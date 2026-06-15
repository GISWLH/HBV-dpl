import numpy as np
import pandas as pd
import torch
import os
# from core.read_configurations import config


# def perv_comp(args, Pptp, Ptc, Infil, Srp, Soil_moist, Smidx_coef, Smidx_exp, Carea_max):
def perv_comp(args, Pptp, Ptc, Infil, Srp, Carea_max, **kwargs):
    # this function is for pervious area computation
    # for srunoff_smidx, we need these inputs as kwargs: Soil_moist, Smidx_coef, Smidx_exp
    # for srunoff_carea, we need Carea_min, Carea_dif, Soil_rechr, Soil_rechr_max
    if args["srunoff_module"] == "srunoff_smidx":
        smidx = Soil_moist + 0.5 * Ptc
        ca_fraction = Smidx_coef * 10 ** (Smidx_exp * smidx)
    elif args["srunoff_module"] == "srunoff_carea":
        ca_fraction = Carea_min + Carea_dif * Soil_rechr / Soil_rechr_max

    # ca_fraction cannot be more than carea_max, no matter which module we are using
    ca_fraction = torch.where(ca_fraction > Carea_max, Carea_max, ca_fraction)
    srpp = ca_fraction * Pptp
    Contrib_fraction = ca_fraction  # not sure why we need this, it is in fortran code srunoff.f90 at line 929. probably we don'tneed it as it is not the output of the subroutine
    Infil = Infil - srpp
    Srp = Srp + srpp

    return Infil, Srp


def check_capacity(Snowinfil_max, Infil, Srp, Soil_moist_max, Soil_moist):
    capacity = Soil_moist_max - Soil_moist
    excess = Infil - capacity

    Srp = torch.where(excess > Snowinfil_max, Srp + excess - Snowinfil_max, Srp)
    Infil = torch.where(excess > Snowinfil_max, Snowinfil_max + capacity, Infil)
    return Infil, Srp


def compute_infil(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp,
    Soil_moist,
    Smidx_coef,
    Smidx_exp,
    hruarea_imperv
):
    NEARZERO = args["NEARZERO"]
    # compute runoff from cascading Hortonian flow  --> we don't have it yet, because there is only one HRU
    # if cascade ==0 --> avail_water = 0.0
    avail_water = torch.zeros(
        Net_rain.shape, dtype=torch.float32, device=args["device"]
    )

    # compute runoff from canopy changeover water
    avail_water = avail_water + Intcp_changeover
    infil = infil + Intcp_changeover

    # pervious area computation
    # this function is for pervious area computation
    # for srunoff_smidx, we need these inputs as kwargs: Soil_moist, Smidx_coef, Smidx_exp
    # for srunoff_carea, we need Carea_min, Carea_dif, Soil_rechr, Soil_rechr_max
    if args["srunoff_module"] == "srunoff_smidx":
        infil_temp, Srp_temp = perv_comp(
            args,
            avail_water,
            avail_water,
            infil,
            Srp,
            Soil_moist,
            Smidx_coef,
            Smidx_exp,
        )
    elif args["srunoff_module"] == "srunoff_carea":
        infil_temp, Srp_temp = perv_comp(
            args,
            avail_water,
            avail_water,
            infil,
            Srp,
            kwargs["Carea_min"],
            kwargs["Carea_dif"],
            kwargs["Soil_rechr"],
            kwargs["Soil_rechr_max"],
        )

    # perv_comp should be activated only for hru_type==1, then:
    infil = torch.where(HRU_type == 1, infil_temp, infil)
    Srp = torch.where(HRU_type == 1, Srp_temp, Srp)

    # pptmix_nopack ==1 means: net_rain > 0 & net_snow > o & pkwater_equiv ==0
    # pptmix_nopack is a Flag indicating that a mixed precipitation event has occurred
    # with no snowpack present on an HRU (1), otherwise (0)
    # if rain/snow event with no antecedent snowpack, compute the runoff from the rain first and then proceed with the
    # snowmelt computations, lines 839 - 847 in srunoff.f90
    avail_water = torch.where(
        (Net_rain > 0) & (Net_snow > 0) & (pkwater_equiv > 0),
        avail_water + Net_rain,
        avail_water,
    )
    infil = torch.where(
        (Net_rain > 0) & (Net_snow > 0) & (pkwater_equiv > 0), infil + Net_rain, infil
    )
    if args["srunoff_module"] == "srunoff_smidx":
        infil_temp, Srp_temp = perv_comp(
            args,
            Net_rain,
            Net_rain,
            infil,
            Srp,
            kwargs["Soil_moist"],
            kwargs["Smidx_coef"],
            kwargs["Smidx_exp"],
        )
    else:
        print("srunoff_carea is not ready yet")
    infil = torch.where(HRU_type == 1, infil_temp, infil)
    Srp = torch.where(HRU_type == 1, Srp_temp, Srp)

    # If precipitation on snowpack, all water available to the surface is
    # considered to be snowmelt, and the snowmelt infiltration
    # procedure is used.  If there is no snowpack and no precip,
    # then check for melt from last of snowpack.  If rain/snow mix
    # with no antecedent snowpack, compute snowmelt portion of runoff
    # Note: probably it should be implemented in snowcomp module.
    avail_water = torch.where(snowmelt > 0.0, avail_water + snowmelt, avail_water)
    infil = torch.where(snowmelt > 0.0, infil + snowmelt, infil)

    # if there is snowmelt and snowpack or )snowmelt and rain or mixed precip). line 859 - 861
    infil_temp, Srp_temp = check_capacity(
        snowinfil_max, infil, Srp, Soil_moist_max, Soil_moist
    )
    infil = torch.where(
        ((snowmelt > 0.0) & (HRU_type == 1) & (pkwater_equiv > 0.0))
        or ((snowmelt > 0.0) & (HRU_type == 1) & ((Net_ppt - Net_snow) < NEARZERO)),
        infil_temp,
        infil,
    )
    Srp = torch.where(
        ((snowmelt > 0.0) & (HRU_type == 1) & (pkwater_equiv > 0.0))
        or ((snowmelt > 0.0) & (HRU_type == 1) & ((Net_ppt - Net_snow) < NEARZERO)),
        Srp_temp,
        Srp,
    )

    # it is the else part in line 864 of srunoff.f90
    # if args["srunoff_module"] == "srunoff_smidx":
    infil_temp, Srp_temp = perv_comp(
        args, snowmelt, Net_ppt, infil, Srp, Soil_moist, Smidx_coef, Smidx_exp
    )
    infil = torch.where(
        ((snowmelt > 0.0) & (HRU_type == 1) & (pkwater_equiv == 0.0))
        or ((snowmelt > 0.0) & (HRU_type == 1) & ((Net_ppt - Net_snow) > NEARZERO)),
        infil_temp,
        infil,
    )
    Srp = torch.where(
        ((snowmelt > 0.0) & (HRU_type == 1) & (pkwater_equiv == 0.0))
        or ((snowmelt > 0.0) & (HRU_type == 1) & ((Net_ppt - Net_snow) > NEARZERO)),
        Srp_temp,
        Srp,
    )

    ### line 871-879 srunoff.f90
    # snowmelt =0 & pkwater_equiv<NEARZERO
    avail_water = torch.where(
        (
            (snowmelt == 0.0)
            & (pkwater_equiv < NEARZERO)
            & (Net_snow < NEARZERO)
            & (Net_rain > 0.0)
        ),
        avail_water + Net_rain,
        avail_water,
    )
    infil = torch.where(
        (
            (snowmelt == 0.0)
            & (pkwater_equiv < NEARZERO)
            & (Net_snow < NEARZERO)
            & (Net_rain > 0.0)
        ),
        infil + Net_rain,
        infil,
    )

    #line 880 srunoff.f90
    infil_temp, Srp_temp = perv_comp(
        args, Net_rain, Net_rain, infil, Srp, Soil_moist, Smidx_coef, Smidx_exp
    )
    infil = torch.where(
        (
        (snowmelt == 0.0)
        & (HRU_type == 1)
        & (pkwater_equiv < NEARZERO)
        & (Net_snow < NEARZERO)
        & (Net_rain > 0.0)
        ),
        infil_temp,
        infil,
    )
    Srp = torch.where(
        (
        (snowmelt == 0.0)
        & (HRU_type == 1)
        & (pkwater_equiv < NEARZERO)
        & (Net_snow < NEARZERO)
        & (Net_rain > 0.0)
        ),
        Srp_temp,
        Srp,
    )


    # lines 887 - 889
    # snowmelt=0.0 & pkwater_equiv >NEARZERO and INFIL > 0.0 and HRU-type=1
    infil_temp, Srp_temp = check_capacity(
        snowinfil_max, infil, Srp, Soil_moist_max, Soil_moist
    )
    infil = torch.where(
        ((snowmelt == 0.0) & (HRU_type == 1) & (pkwater_equiv >= NEARZERO) & (infil > 0.0)),
        infil_temp,
        infil,
    )
    Srp = torch.where(
        ((snowmelt == 0.0) & (HRU_type == 1) & (pkwater_equiv >= NEARZERO) & (infil > 0.0)),
        Srp_temp,
        Srp,
    )


    # lines 891 - 900 srunoff.f90
    # impervious area computation
    Imperv_stor = torch.where(hruarea_imperv > 0.0,
                              Imperv_stor + avail_water,
                              Imperv_stor)
    Sri = torch.where((hruarea_imperv > 0.0)
                      & (HRU_type == 1)
                      & (Imperv_stor > Imperv_stor_max),
                      Imperv_stor - Imperv_stor_max,
                      torch.zeros(Imperv_stor.shape, dtype=torch.float32, device=args["device"]))
    Imperv_stor = torch.where((hruarea_imperv > 0.0)
                      & (HRU_type == 1)
                      & (Imperv_stor > Imperv_stor_max),
                      Imperv_stor_max,
                      Imperv_stor)






    return infil, Srp, Sri, Imperv_stor


def srunoff_carea(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp,
    Carea_min,
    Carea_dif,
    Soil_rechr,
    Soil_rechr_max,
):
    infil, Srp, Sri, Imperv_stor = compute_infil(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp,
    Soil_moist,
    Smidx_coef,
    Smidx_exp,
    hruarea_imperv,
        Carea_dif,
        Soil_rechr,
        Soil_rechr_max,
    )

    return retention_storage, evaporation, runoff_imperv, infil, Srp

def dprst_comp(args, Net_rain, snowmelt, pkwater_equiv, Net_snow, Dprst_in, harea,
    Dprst_area_open_max,
    Dprst_area_clos_max,
    Dprst_vol_open_max,
    Dprst_vol_clos_max,Dprst_vol_open ,Dprst_vol_clos,  Srp, hruarea_imperv,
    hruarea_perv, Sri,):
    # Dprst_vol_open:    Storage volume in open surface depressions for each HRU (acre-inches in fortran code)
    # dprst_vol_clos:  Storage volume in closed surface depressions for each HRU/(acre-inches in fortran code)
    NEARZERO = args["NEARZERO"]
    # because cascade_flag == 0 in our case
    inflow = torch.zeros(torch.zeros(Net_rain.shape, dtype=torch.float32,
                               device=args["device"]))

    # check pptmix_nopack in snowcomp module. it should come from that module
    pptmix_nopack = torch.where((Net_rain > 0) & (Net_snow > 0) & (pkwater_equiv > 0),
                                torch.ones(Net_rain.shape, dtype=torch.float32, device=args["device"]),
                                torch.zeros(Net_rain.shape, dtype=torch.float32, device=args["device"]))
    inflow = torch.where(Pptmix_nopack == 1,
                         inflow + Net_rain,
                         inflow)
    # If precipitation on snowpack all water available to the surface is considered to be snowmelt
    # If there is no snowpack and no precip,then check for melt from last of snowpack.
    # If rain/snow mix with no antecedent snowpack, compute snowmelt portion of runoff.
    # lines 1203 - 1215 srunoff.f90
    inflow = torch.where(snowmelt > 0.0,
                         inflow + snowmelt,
                         inflow)
    inflow = torch.where((snowmelt == 0.0)
                         & (pkwater_equiv < NEARZERO)
                         & (Net_snow < NEARZERO)
                         & (Net_rain > 0.0),
                         inflow + Net_rain)


    ### line 1217 of srunoff.f90
    # Farshid: Dprst_in is already zero when it comes to this function. so we don't need the following line?
    # we need to see why it is zero
    # Dprst_in = torch.zeros(Net_rain.shape, dtype=torch.float32, vice=args["device"])
    #open
    Dprst_in = torch.where(Dprst_area_open_max > 0.0,
                           inflow * Dprst_area_open_max,
                           Dprst_in)
    Dprst_vol_open = torch.where(Dprst_area_open_max > 0.0,
                           Dprst_vol_open + Dprst_in,
                           Dprst_vol_open)
    #clos
    tmp1 = torch.where(Dprst_area_clos_max > 0.0,
                           inflow * Dprst_area_clos_max,
                           torch.zeros(inflow.shape, dtype=torch.float32, device=args["device"]))
    Dprst_vol_clos = torch.where(Dprst_area_clos_max > 0.0,
                                 Dprst_vol_clos + tmp1,
                                 Dprst_vol_clos)
    Dprst_in = torch.where(Dprst_area_clos_max > 0.0,
                           Dprst_in + tmp1,
                           Dprst_in)
    #line 1227 srunoff.f90
    Dprst_in = Dprst_in / harea             # inches/hru in fortran code
    # add any pervious surface runoff fraction to depressions, lines 1229-1231
    dprst_srp = torch.zeros(Dprst_in.shape, dtype=torch.float32, device=args["device"])
    dprst_sri = torch.zeros(Dprst_in.shape, dtype=torch.float32, device=args["device"])

    # line 1232 - 1250 srunoff.f90
    Perv_frac = hruarea_perv / harea
    Perv_frac = torch.clamp(Perv_frac, min=0.0, max=1.0)
    tmp = torch.where(Srp > 0.0,
                      Srp * Perv_frac * sro_to_dprst_perv * harea,
                      torch.zeros(Dprst_in.shape, dtype=torch.float32, device=args["device"]))
    dprst_srp_open
    dprst_srp
    Dprst_vol_open

    return

def srunoff_smidx(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp,
    Soil_moist,
    Smidx_coef,
    Smidx_exp,
    hruarea_imperv,
    hruarea_perv,
    Dprst_area_max,
    harea,
    Dprst_area_open_max,
    Dprst_area_clos_max,
    Dprst_vol_open_max,
    Dprst_vol_clos_max,Dprst_vol_open ,Dprst_vol_clos
):
    # if Use_sroff_transfer==1 --> we don't have this yet. It is about reading transfer flow rate from external files

    # compute infiltration
    # Srp: water available for infiltration
    infil, Srp, Sri, Imperv_stor = compute_infil(
    args,
    Net_rain,
    Net_ppt,
    Imperv_stor,
    Imperv_stor_max,
    snowmelt,
    snowinfil_max,
    Net_snow,
    pkwater_equiv,
    infil,
    HRU_type,
    Intcp_changeover,
    Srp,
    Soil_moist,
    Smidx_coef,
    Smidx_exp,
    hruarea_imperv,

    )

    #### lines 669 - 683 of srunoff.f90
    # Dprst_flag = 1 means depression storage simulation is computed ( 0= no, 1 = yes)
    if args["Dprst_flag"] == 1:
        Dprst_flag = torch.ones(Net_rain.shape, dtype=torch.float32, device=args["device"])
        Dprst_in = torch.zeros(Net_rain.shape, dtype=torch.float32, device=args["device"])  # if I run srunoffinit, this should come from that module
        dprst_chk = torch.zeros(Net_rain.shape, dtype=torch.float32,
                               device=args["device"])  # if I run srunoffinit, this should come
        dprst_chk = torch.where(Dprst_area_max > 0.0,
                                torch.ones(dprst_chk.shape, dtype=torch.float32, device=args["device"]),
                                dprst_chk)

        #compute the depression storage component
        # only call if total depression surface area for each hru is > 0.0
        #availh2o is a local variable
        availh2o = torch.where(Dprst_area_max > 0.0,
                               Intcp_changeover + Net_rain,
                               availh2o)
        # computing depression
        Dprst_sroff_hru = dprst_comp()

    elif args["Dprst_flag"] == 0:
        Dprst_flag = torch.zeros(Net_rain.shape, dtype=torch.float32, device=args["device"])



    return retention_storage, evaporation, runoff_imperv, infil, Srp


def srunoff(args, **kwargs):
    # probably we need to run srunoffinit() first in the beginning of warm-up!


    # sroff_flag==1 in fortran code
    if args["srunoff_module"] == "srunoff_smidx":
        retention_storage, evaporation, runoff_imperv, infil, Srp = srunoff_smidx(
            args,
            kwargs["Net_rain"],
            kwargs["Net_ppt"],
            kwargs["Imperv_stor"],
            kwargs["Imperv_stor_max"],
            kwargs["snowmelt"],
            kwargs["snowinfil_max"],
            kwargs["Net_snow"],
            kwargs["pkwater_equiv"],
            kwargs["infil"],
            kwargs["HRU_type"],
            kwargs["Intcp_changeover"],
            kwargs["Srp"],
            kwargs["Soil_moist"],
            kwargs["Smidx_coef"],
            kwargs["Smidx_exp"],
            kwargs["hruarea_imperv"],
            kwargs["hruarea_perv"],
            kwargs["Dprst_area_max"],
            kwargs["harea"],
            kwargs["Dprst_area_open_max"],
            kwargs["Dprst_area_clos_max"],
            kwargs["Dprst_vol_open_max"],
            kwargs["Dprst_vol_clos_max"],
            kwargs["Dprst_vol_open"],
            kwargs ["Dprst_vol_clos"]

        )

    # sroff_flag=2 in fortran code
    elif args["srunoff_module"] == "srunoff_carea":
        print("this surface runoff method is not ready yet")
        retention_storage, evaporation, runoff_imperv, infil, Srp = srunoff_carea(
            args,
            kwargs["Net_rain"],
            kwargs["Net_ppt"],
            kwargs["Imperv_stor"],
            kwargs["Imperv_stor_max"],
            kwargs["snowmelt"],
            kwargs["snowinfil_max"],
            kwargs["Net_snow"],
            kwargs["pkwater_equiv"],
            kwargs["infil"],
            kwargs["HRU_type"],
            kwargs["Intcp_changeover"],
            kwargs["Srp"],
            kwargs["Soil_moist"],
            kwargs["Smidx_coef"],
            kwargs["Smidx_exp"],
            kwargs["Dprst_area_max"],
        )
    return retention_storage, evaporation, runoff_imperv, infil, Srp
