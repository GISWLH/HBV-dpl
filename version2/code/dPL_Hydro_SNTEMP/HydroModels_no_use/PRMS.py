import torch
from MODELS.PET_models.potet import get_potet
from HydroModels.surface_runoff import srunoff


class PRMS_pytorch(torch.nn.Module):
    def __init__(self):
        super(PRMS_pytorch, self).__init__()
        self.srunoff = srunoff

    def precip_form(
        self,
        Precip,
        Tmaxf,
        Tminf,
        Tmax_allsnow_f,
        Tmax_allrain_offset,
        args,
        Adjmix_rain=1.0,
        NEARZERO=0.00001,
        rain_adj=1.0,
        snow_adj=1.0,
    ):
        """

        :param Precip: HRU Precipitation
        :param Tmaxf: max air temp
        :param Tminf: min air temp
        :param Tmax_allsnow_f: Monthly (January to December) maximum air temperature when precipitation is assumed to
                            be snow; if HRU air temperature is less than or equal to this value, precipitation is snow
        :param Tmax_allrain_offset: Monthly (January to December) maximum air temperature when precipitation is assumed to
                        be rain; if HRU air temperature is greater than or equal to Tmax_allsnow_f plus this value,
                        precipitation is rain
        :param Adjmix_rain: Monthly (January to December) factor to adjust rain proportion in
                        a mixed rain/snow event  (0.6 - 1.4)
        :param NEARZERO:
        :param rain_adj: Monthly (January to December) factor to adjust measured precipitation on each HRU to
                        account for differences in elevation, and so forth (0.5 - 2.0)
        :param snow_adj: Monthly (January to December) factor to adjust measured precipitation on each HRU to
                        account for differences in elevation, and so forth (0.5 - 2.0)
        :return: Basin area-weighted average rainfall,
                Basin area-weighted average snowfall
        """
        # if max temp is below or equal to the base temp for snow then
        # precipitation is all snow
        snow_hru = snow_adj * torch.where(Tmaxf <= Tmax_allsnow_f, Precip, Precip * 0.0)

        mask_snow_hru = snow_hru.le(0.0).float()  # shows when it is not snowing
        # if min temp is above base temp for snow or
        # max temp is above all_rain temp then the precipitation is all rain

        rain_hru = rain_adj * torch.where(
            (Tminf > Tmax_allsnow_f) | (Tmaxf >= Tmax_allsnow_f + Tmax_allrain_offset),
            Precip * mask_snow_hru,
            Precip * mask_snow_hru * 0.0,
        )
        mask_rain_hru = rain_hru.le(0.0).float()
        # otherwise precipitation is a mixture of rain and snow
        tdiff = Tmaxf - Tminf
        if tdiff.le(-NEARZERO).int().sum() > 0.0:  # Tmax < Tmin
            print("ERROR, tmax < tmin ")

        tdiff_min = torch.zeros(tdiff.shape, device=args["device"]) + 0.0001
        tdiff = torch.where(tdiff < NEARZERO, tdiff_min, tdiff)
        Prmx = ((Tmaxf - Tmax_allsnow_f) / tdiff) * Adjmix_rain
        Prmx = torch.clamp(Prmx, min=0.0, max=1.0)
        hru_ppt = Precip * snow_adj * mask_rain_hru * mask_snow_hru
        rain_hru_mix = hru_ppt * Prmx
        snow_hru_mix = hru_ppt - rain_hru_mix

        Basin_rain = rain_hru + rain_hru_mix
        Basin_snow = snow_hru + snow_hru_mix

        return Basin_rain, Basin_snow

    def PRMS_transp_month_ON_OFF(self, months, args, transp_beg=None, transp_end=None):

        # here if we don't define the ON OFF time for transp, it is assumed to be started at month3 and ended at month 9
        if transp_beg == None:
            transp_beg = (
                torch.ones(months.shape, dtype=torch.float32, device=args["device"])
                + 2.0
            )
        if transp_end == None:
            transp_end = (
                torch.ones(months.shape, dtype=torch.float32, device=args["device"])
                + 8.0
            )

        # this mask tells us when the transpiration is ON
        mask_transp_ON = torch.where(
            (months >= transp_beg) & (months < transp_end),
            torch.ones(months.shape, device=args["device"], dtype=torch.float32),
            torch.zeros(months.shape, device=args["device"], dtype=torch.float32),
        )
        return mask_transp_ON

    def PRMS_transp_tindex(
        self, tmaxf, args, months, transp_beg=None, transp_end=None, transp_tmax=None
    ):

        # getting the time that transp starts (0 or 1)
        mask_transp_month_ON = self.PRMS_transp_month_ON_OFF(
            months, args, transp_beg=transp_beg, transp_end=transp_end
        )

        # transp does not occurs in freezing temperature
        tmaxf_pos_mask = torch.where(
            tmaxf > 0.0,
            torch.ones(tmaxf.shape, device=args["device"], dtype=torch.float32),
            torch.zeros(tmaxf.shape, device=args["device"], dtype=torch.float32),
        )
        tmax_mod = tmaxf * tmaxf_pos_mask * mask_transp_month_ON
        Tmax_sum = torch.cumsum(
            tmax_mod, axis=1
        )  # tmax_mod.shape[basin, rho + bufftime, nmul]

        if transp_tmax == None:
            transp_tmax = (
                torch.ones(Tmax_sum.shape, device=args["device"], dtype=torch.float32)
                * 149.0
            )  # we assume it is 300 in F and 149 in C

        mask_transp_tmax = torch.where(
            Tmax_sum >= transp_tmax,
            torch.ones(Tmax_sum.shape, device=args["device"], dtype=torch.float32),
            torch.zeros(Tmax_sum.shape, device=args["device"], dtype=torch.float32),
        )

        # A mask that shows when transp is ON (it should be between transp_beg and transp_end
        # and also reaches the transp_tmax as well)
        Basin_transp_on = mask_transp_tmax * mask_transp_month_ON
        return Basin_transp_on

    def intercept(self, Precip, Stor_max, Cov, intcp_stor):
        Net_Precip = Precip * (1 - Cov)
        intcp_stor = intcp_stor + Precip
        Net_Precip = torch.where(
            intcp_stor > Stor_max,
            Net_Precip + (intcp_stor - Stor_max) * Cov,
            Net_Precip,
        )
        intcp_stor = torch.where(intcp_stor > Stor_max, Stor_max, intcp_stor)
        return Net_Precip, intcp_stor

    def intcp(self, args, c_PRMS, Hru_rain, Hru_snow, Basin_transp_on, intcp_stor):
        # why do we have these in the fortran code? intcp.f90 lines 310 - 314
        # IF (Transp_on(i) == 1)
        # THEN
        # Canopy_covden(i) = Covden_sum(i)
        # ELSE
        # Canopy_covden(i) = Covden_win(i)
        # ENDIF
        # if transp_On == 1 --> canopy_covden = covden_sum
        nmul = args["nmul"]
        covden_sum = (
            c_PRMS[:, (args["optData"]["varC_PRMS"]).index("covden_sum")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )
        covden_win = (
            c_PRMS[:, (args["optData"]["varC_PRMS"]).index("covden_win")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )
        srain_intcp_max = (
            c_PRMS[:, (args["optData"]["varC_PRMS"]).index("srain_intcp(mm)")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )
        snow_intcp_max = (
            c_PRMS[:, (args["optData"]["varC_PRMS"]).index("snow_intcp(mm)")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )
        wrain_intcp_max = (
            c_PRMS[:, (args["optData"]["varC_PRMS"]).index("wrain_intcp(mm)")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )
        cov_type = (
            c_PRMS[:, (args["optData"]["varC_PRMS"]).index("cov_type")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )

        #  translation of lines 313-340 of intcp.f90, adjusment interception amounts in summer and winter
        # vegetation cover density for the major vegetation type (it is a fraction)
        # if transpiration is ON, it s considered as summer!
        cov = torch.where(Basin_transp_on == 1.0, covden_sum, covden_win)
        intcp_form = torch.where(
            Hru_snow > 0.0,
            torch.ones(Hru_snow.shape, device=args["device"], dtype=torch.float32),
            torch.zeros(Hru_snow.shape, device=args["device"], dtype=torch.float32),
        )
        # cov_type=0 --> bare ground --> no intcp storage
        # it should be done for lake area/hrus too, now I don't have any lake
        extrawater = torch.where(
            (cov_type == 0.0) & (intcp_stor > 0.0),
            intcp_stor,
            torch.zeros(cov_type.shape, device=args["device"], dtype=torch.float32),
        )
        # adjusting intcp storage for bare grounds. The unit is (L (length))
        intcp_stor = torch.where(
            (cov_type == 0.0) & (intcp_stor > 0.0),
            torch.zeros(intcp_stor.shape, device=args["device"], dtype=torch.float32),
            intcp_stor,
        )

        # ###Determine the amount of interception from rain
        # go from summer to winter.   translation of  lines 344 - 362 intcp.f90
        #  Farshid: what happens if cov > covden_sum? Answer: It doesn't happen n current dataset
        diff = torch.where(
            (Basin_transp_on == 0.0) & (intcp_stor > 0.0),
            covden_sum - cov,
            torch.zeros(
                Basin_transp_on.shape, device=args["device"], dtype=torch.float32
            ),
        )
        # changeover = torch.where((Basin_transp_on == 0.0) & (intcp_stor > 0.0),
        #                    diff * intcp_stor,
        #                    torch.zeros(Basin_transp_on.shape, device=args["device"], dtype=torch.float32))
        changeover = (
            diff * intcp_stor
        )  # seems this line is the same as the upper three lines
        intcp_stor = torch.where(
            (cov > 0.0)
            & (changeover < 0.0)
            & (Basin_transp_on == 0.0)
            & (intcp_stor > 0.0),
            intcp_stor * covden_sum / cov,
            intcp_stor,
        )
        changeover = torch.where(
            (cov > 0.0)
            & (changeover < 0.0)
            & (Basin_transp_on == 0.0)
            & (intcp_stor > 0.0),
            torch.zeros(changeover.shape, device=args["device"], dtype=torch.float32),
            changeover,
        )
        intcp_stor = torch.where(
            (Basin_transp_on == 0.0) & (intcp_stor > 0.0) & (cov <= 0.0),
            torch.zeros(intcp_stor.shape, device=args["device"], dtype=torch.float32),
            intcp_stor,
        )

        ##  go from winter to summer.   translation of  lines 365 - 382 intcp.f9
        diff = torch.where(
            (Basin_transp_on == 1.0) & (intcp_stor > 0.0), covden_win - cov, diff
        )
        changeover = torch.where(
            (Basin_transp_on == 1.0) & (intcp_stor > 0.0), diff * intcp_stor, changeover
        )
        intcp_stor = torch.where(
            (cov > 0.0)
            & (changeover < 0.0)
            & (Basin_transp_on == 1.0)
            & (intcp_stor > 0.0),
            intcp_stor * covden_win / cov,
            intcp_stor,
        )
        changeover = torch.where(
            (cov > 0.0)
            & (changeover < 0.0)
            & (Basin_transp_on == 1.0)
            & (intcp_stor > 0.0),
            torch.zeros(changeover.shape, device=args["device"], dtype=torch.float32),
            changeover,
        )
        intcp_stor = torch.where(
            (Basin_transp_on == 1.0) & (intcp_stor > 0.0) & (cov <= 0.0),
            torch.zeros(intcp_stor.shape, device=args["device"], dtype=torch.float32),
            intcp_stor,
        )

        ## determine the amount of interception from rain translation lines 386 - 410 of intcp.f90
        # stor = stor_Max
        # for rain, different values for winter and summer
        stor = torch.where(
            (cov_type != 0.0) & (Basin_transp_on == 1.0),
            srain_intcp_max,
            torch.zeros(
                srain_intcp_max.shape, device=args["device"], dtype=torch.float32
            ),
        )
        stor = torch.where(
            (cov_type != 0.0) & (Basin_transp_on == 0.0), wrain_intcp_max, stor
        )

        # it should be done for cov_type>1 , bare_soil = 0, grass=1, shrub = 2, tree = 3, conifereous=4
        intcp_rain_temp, intcp_stor_temp = self.intercept(
            Hru_rain, stor, cov, intcp_stor
        )
        # grass and bare soil cannot hold rain ==> interception = 0
        intcp_rain = torch.where(cov_type > 1, intcp_rain_temp, Hru_rain)
        intcp_stor = torch.where(cov_type > 1, intcp_stor_temp, intcp_stor)

        # I didnot code the irrigation part  , lines 415-435 intcp.f90

        #  determine the amount of interception from snow
        # for snow, one value for all seasons
        intcp_snow_temp, intcp_stor_temp = self.intercept(
            Hru_snow, snow_intcp_max, cov, intcp_stor
        )
        # bare soil cannot hold snow, grass can!
        intcp_snow = torch.where(cov_type > 0, intcp_snow_temp, Hru_snow)
        intcp_stor = torch.where(cov_type > 0, intcp_snow_temp, intcp_stor)

        return intcp_rain, intcp_snow, intcp_stor, changeover

        # intcp_form==1 if Hru_snow>0

    def forward(self, x, c_PRMS, params, args, warm_up=0, init=False):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args["optData"]["varT_PRMS"]
        vars_c_PRMS = args["optData"]["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, 0:warm_up, :]
                warm_up_model = PRMS_pytorch()
                (
                    intcpstor,
                    infil,
                    Imperv_stor,
                    pkwater_equiv,
                    MELTWATER,
                    Srp,
                    Soil_moist,
                    Dprst_vol_open,
                    Dprst_vol_clos
                ) = warm_up_model(xinit, c_PRMS, paramsinit, args, warm_up=0, init=True)
        else:
            # zero for initializiation [No_basins, nmul]
            intcpstor = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            infil = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            Imperv_stor = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            pkwater_equiv = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )  # (SNOWPACK in HBV)
            MELTWATER = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            # Srp: water available for infiltration
            Srp = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            Soil_moist = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            # Storage volume in open surface depressions for each HRU
            Dprst_vol_open = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )
            Dprst_vol_clos = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            )

        # potet_sublim can be one of the parameters. between 0.1-0.75
        potet_sublim = params[:, warm_up:, 0:nmul]
        hamon_coef = params[:, warm_up:, nmul : nmul * 2]
        parCFmax = params[:, warm_up:, 2 * nmul : nmul * 3]
        # x_new = x[:, warm_up:, :]
        # Ngrid, Ndays = x_new.shape[0], x_new.shape[1]
        Precip = (
            x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        month = x[:, warm_up:, vars.index("month")].unsqueeze(-1).repeat(1, 1, nmul)
        dayl = x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        mean_air_temp = (Tmaxf + Tminf) / 2
        Ngrid, Ndays = Precip.shape[0], Precip.shape[1]

        # attributes. [no_basins, nmul]
        harea = c_PRMS[:, vars_c_PRMS.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1, nmul)
        hru_percent_imperv = (
            c_PRMS[:, vars_c_PRMS.index("hru_percent_imperv")]
            .unsqueeze(-1)
            .repeat(1, nmul)
        )
        Smidx_coef = (
            c_PRMS[:, vars_c_PRMS.index("Smidx_coef")].unsqueeze(-1).repeat(1, nmul)
        )
        Smidx_exp = (
            c_PRMS[:, vars_c_PRMS.index("Smidx_exp")].unsqueeze(-1).repeat(1, nmul)
        )
        hruarea_imperv = hru_percent_imperv * harea   # hruarea_imperv = hru_imperv
        hruarea_perv = harea - hruarea_imperv      #  hruarea_perv = hru_perv
        #Dprst_frac: Fraction of each HRU area that has surface depressions
        Dprst_frac = (
            c_PRMS[:, vars_c_PRMS.index("dprst_frac")].unsqueeze(-1).repeat(1, nmul)
        )
        Dprst_frac = torch.clamp(Dprst_frac, min=0.0, max=1.0)  # just to make sure it is in the range
        Dprst_area_max = Dprst_frac * harea

        Dprst_frac_open = (
            c_PRMS[:, vars_c_PRMS.index("dprst_frac_open")].unsqueeze(-1).repeat(1, nmul)
        )
        # from lines 396 - 399 basin.f90
        Dprst_area_open_max = torch.where(Dprst_area_max > 0.0,
                                          Dprst_area_max * Dprst_frac_open,
                                          torch.zeros(Dprst_frac_open.shape, dtype=torch.float32, device=args["device"]))
        Dprst_area_clos_max = torch.where(Dprst_area_max > 0.0,
                                          Dprst_area_max - Dprst_area_open_max,
                                          torch.zeros(Dprst_frac_open.shape, dtype=torch.float32,
                                                      device=args["device"]))
        Dprst_frac_clos = torch.where(Dprst_area_max > 0.0,
                                          1.0 - Dprst_frac_open,
                                          torch.zeros(Dprst_frac_open.shape, dtype=torch.float32,
                                                      device=args["device"]))

        # from lines1092-1095 of srunoff.f90
        Dprst_depth_avg = (
            c_PRMS[:, vars_c_PRMS.index("dprst_depth_avg")].unsqueeze(-1).repeat(1, nmul)
        )
        Dprst_vol_open_max = Dprst_area_open_max * Dprst_depth_avg
        Dprst_vol_clos_max = Dprst_area_clos_max * Dprst_depth_avg

        # snow and rain temperature thresholds
        Tmax_allsnow_f = torch.zeros(
            Tmaxf.shape, dtype=torch.float32, device=args["device"]
        )
        Tmax_allrain_offset = (
            torch.ones(Tmaxf.shape, dtype=torch.float32, device=args["device"]) + 0.5
        )

        # snow or rain
        Basin_rain, Basin_snow = self.precip_form(
            Precip,
            Tmaxf,
            Tminf,
            Tmax_allsnow_f,
            Tmax_allrain_offset,
            args,
            Adjmix_rain=1.0,
            NEARZERO=NEARZERO,
            rain_adj=1.0,
            snow_adj=1.0,
        )
        # transpiration module, if there were multiple methods, the method should be specified
        # in config file, like what I did for PET module.
        Basin_transp_on = self.PRMS_transp_tindex(
            Tmaxf,
            args,
            months=month,
            transp_beg=None,
            transp_end=None,
            transp_tmax=None,
        )

        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        )

        # initialize the Q_sim
        Q_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])

        for t in range(Ndays):
            Intcp_transp_on = Basin_transp_on[:, t, :]

            # daily precip
            Precip_day = Precip[:, t, :]
            rain_day = Basin_rain[:, t, :]
            snow_day = Basin_snow[:, t, :]
            PET_day = PET[:, t, :]
            # for snowmelt module from HBV. We need to substitute it for the PRMS snowcomp in future
            Tmax_allsnow_f_day = Tmax_allsnow_f[:, t, :]
            Tmaxf_day = Tmaxf[:, t, :]
            potet_sublim_day = potet_sublim[:, t, :]
            # interception
            intcp_rain, intcp_snow, intcpstor, Intcp_changeover = self.intcp(
                args, c_PRMS, rain_day, snow_day, Intcp_transp_on, intcpstor
            )

            # evaporation or sublimation of interception, lines 482 - 518 intcp.f90
            # if precipitation happens --> no evaporation or sublimation
            # not sure if we need to have Epan_coef in params
            # evaporation
            evrn = torch.where(
                (Precip_day < NEARZERO) & (intcpstor > 0.0),
                PET_day / args["Epan_coef"],
                torch.zeros(
                    intcpstor.shape, dtype=torch.float32, device=args["device"]
                ),
            )
            # sublimation
            evsn = torch.where(
                (Precip_day < NEARZERO) & (intcpstor > 0.0),
                PET_day * potet_sublim_day,
                torch.zeros(
                    intcpstor.shape, dtype=torch.float32, device=args["device"]
                ),
            )

            # for snow days, we subtract the sublimation
            intcpstor = torch.where((snow_day > 0.0), intcpstor - evsn, intcpstor)
            intcpstor = torch.where(
                (intcpstor < 0.0) & (snow_day > 0.0),
                torch.zeros(
                    intcpstor.shape, dtype=torch.float32, device=args["device"]
                ),
                intcpstor,
            )
            # elif rainy days
            intcpstor = torch.where((snow_day <= 0.0), intcpstor - evrn, intcpstor)
            intcpstor = torch.where(
                (intcpstor < 0.0) & (snow_day <= 0.0),
                torch.zeros(
                    intcpstor.shape, dtype=torch.float32, device=args["device"]
                ),
                intcpstor,
            )

            # snow module. currently from HBV (SNOWPACK=pkwater_equiv), (melt=snowmelt)
            # pkwater_equiv: snow-water equivalent storage in the snowpack
            pkwater_equiv = pkwater_equiv + snow_day
            snowmelt = parCFmax * (Tmax_allsnow_f - Tmax_allsnow_f_day)
            snowmelt = torch.clamp(snowmelt, min=0.0)
            snowmelt = torch.min(snowmelt, pkwater_equiv)
            MELTWATER = MELTWATER + snowmelt
            pkwater_equiv = pkwater_equiv - snowmelt

            # surface runoff module
            # Imperv_stor : Storage on impervious area for each HRU
            # pkwater_equiv: snow pack water equicalent on each HRU
            # infil: infiltration to the capilary reservoir for each HRU
            #
            # We can read this HRU_type fom PRMS NHM parameters excel files.
            # HRU_types (0:inactive, 1:land, 2:lake, 3:swale)
            HRU_type = torch.ones(
                rain_day.shape, dtype=torch.float32, device=args["device"]
            )

            retention_storage, evaporation, runoff_imperv, infil, Srp = self.srunoff(
                args=args,
                Net_rain=rain_day,
                Net_ppt=Precip_day,
                Imperv_Stor=Imperv_stor,
                Intcp_changeover=Intcp_changeover,
                Imperv_stor_max=0.05
                * 0.0254,  # 0.05 inches constant for all --> converted to mm
                snowmelt=0.01,
                snowinfil_max=2.0 * 25.4,  # based on tm6b9 --> converted to mm
                Net_snow=snow_day,
                pkwater_equiv=pkwater_equiv,
                infil=infil,
                HRU_type=HRU_type,
                Srp=Srp,
                Soil_moist=Soil_moist,
                Smidx_coef=Smidx_coef,
                Smidx_exp=Smidx_exp,
                hruarea_imperv=hruarea_imperv,
                hruarea_perv=hruarea_perv,
                Dprst_area_max=Dprst_area_max,
                harea=harea,
                Dprst_area_open_max=Dprst_area_open_max,
                Dprst_area_cllos_max=Dprst_area_clos_max,
                Dprst_vol_open_max=Dprst_vol_open_max,
                Dprst_vol_clos_max=Dprst_vol_clos_max,
                Dprst_vol_open=Dprst_vol_open,
                Dprst_vol_clos=Dprst_vol_clos
            )

            Q_sim[:, t, :] = intcpstor

            # print("END")
        if init == True:
            return (
                intcpstor,
                infil,
                Imperv_stor,
                pkwater_equiv,
                MELTWATER,
                Srp,
                Soil_moist,
                Dprst_vol_open,
                Dprst_vol_clos
            )
        elif init == False:
            return Q_sim * PET


class srunoff(torch.nn.Module):
    def __init__(self):
        super(srunoff, self).__init__()


# Precip = torch.rand((5, 10, 15))
# Tmax_allrain_f = torch.rand((5, 10, 15)) + 5.0
# Tmax_allsnow_f = torch.rand((5, 10, 15))
# Tmaxf = torch.rand((5, 10, 15)) + 10.0
# Tmaxf[0,0, :] = - 5.0
# Tminf = torch.rand((5, 10, 15))
# Tminf[0, 0, :] = - 15.0
# Tminf[0,2, :] = Tmax_allsnow_f[0,2, :] - 3.0
# Tmaxf[0,2, :] = Tmax_allrain_f[0,2, :] - 1.0
# basin_rain, Basin_snow = precip_form(self, Precip, Tmaxf, Tminf, Tmax_allsnow_f, Tmax_allrain_f)
#
#
#
# months = torch.randint(1, 12, Tminf.shape)
# Basin_transp_on = PRMS_transp_tindex(Tmaxf, months, transp_beg=None, transp_end=None, transp_tmax=None)
