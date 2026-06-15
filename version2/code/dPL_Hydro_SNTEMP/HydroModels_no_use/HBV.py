import torch
import torch.nn as nn

class HBV(nn.Module):
    def __init__(self, theta,delta_t,climate_data):
        super().__init__()

        self.register_parameter("theta", theta)
        self.delta_t = delta_t
        self.climate_data = climate_data

    def forward(self, t, y):
        ##parameters
        tt      = self.theta[0];    # % TT, middle of snow-rain interval [oC]
        tti     = self.theta[1];    # % TTI, interval length of rain-snow spectrum [oC]
        ttm     = self.theta[2];    # % TTM, threshold temperature for snowmelt [oC]
        cfr     = self.theta[3];    # % CFR, coefficient of refreezing of melted snow [-]
        cfmax   = self.theta[4];    # % CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
        whc     = self.theta[5];    # % WHC, maximum water holding content of snow pack [-]
        cflux   = self.theta[6];   #	% CFLUX, maximum rate of capillary rise [mm/d]
        fc      = self.theta[7];    # % FC, maximum soil moisture storage [mm]
        lp      = self.theta[8];    # % LP, wilting point as fraction of FC [-]
        beta    = self.theta[9];   # % BETA, non-linearity coefficient of upper zone recharge [-]
        k0      = self.theta[10];   # % K0, runoff coefficient from upper zone [d-1],
        alpha   = self.theta[11];   # % ALPHA, non-linearity coefficient of runoff from upper zone [-]
        perc    = self.theta[12];   # % PERC, maximum rate of percolation to lower zone [mm/d]
        k1      = self.theta[13];   # % K1, runoff coefficient from lower zone [d-1]
        # maxbas  = self.theta[15];   # % MAXBAS, flow routing delay [d]

        ##% stores
        S1 = y[:,0].clone();
        S2 = y[:,1].clone();
        S3 = y[:,2].clone();
        S4 = y[:,3].clone();
        S5 = y[:,4].clone();
        dS = torch.zeros(y.shape).to(y)
        fluxes = torch.zeros((y.shape[0],12)).to(y)

        climate_in = self.climate_data[int(t),:,:];   ##% climate at this step
        P  = climate_in[:,0];
        Ep = climate_in[:,1];
        T  = climate_in[:,2];

        ##% fluxes functions

        flux_sf   = self.snowfall_2(P,T,tt,tti);
        flux_refr = self.refreeze_1(cfr,cfmax,ttm,T,S2,self.delta_t);
        flux_melt = self.melt_1(cfmax,ttm,T,S1,self.delta_t);
        flux_rf   = self.rainfall_2(P,T,tt,tti);
        flux_in   = self.infiltration_3(flux_rf+flux_melt,S2,whc*S1);
        flux_se   = self.excess_1(S2,whc*S1,self.delta_t);
        flux_cf   = self.capillary_1(cflux,S3,fc,S4,self.delta_t);
        flux_ea   = self.evap_3(lp,S3,fc,Ep,self.delta_t);
        flux_r    = self.recharge_2(beta,S3,fc,flux_in+flux_se);
        flux_q0   = self.interflow_2(k0,S4,alpha,self.delta_t);
        flux_perc = self.percolation_1(perc,S4,self.delta_t);
        flux_q1   = self.baseflow_1(k1,S5);
        #flux_qt   = route(flux_q0 + flux_q1, uh);

        #% stores ODEs
        dS[:,0] = flux_sf   + flux_refr - flux_melt;
        dS[:,1] = flux_rf   + flux_melt - flux_refr - flux_in - flux_se;
        dS[:,2] = flux_in   + flux_se   + flux_cf   - flux_ea - flux_r;
        dS[:,3] = flux_r    - flux_cf   - flux_q0   - flux_perc;
        dS[:,4] = flux_perc - flux_q1;

        fluxes[:,0] =flux_sf
        fluxes[:,1] =flux_refr
        fluxes[:,2] =flux_melt
        fluxes[:,3] =flux_rf
        fluxes[:,4] =flux_in
        fluxes[:,5] =flux_se
        fluxes[:,6] =flux_cf
        fluxes[:,7] =flux_ea
        fluxes[:,8] =flux_r
        fluxes[:,9] =flux_q0
        fluxes[:,10] =flux_perc
        fluxes[:,11] =flux_q1


        return dS,fluxes

    def snowfall_2(self,In,T,p1,p2):
        return torch.minimum(In,torch.maximum(torch.tensor(0),In*(p1+0.5*p2-T)/p2));

    def refreeze_1(self,p1,p2,p3,T,S,dt):
        return torch.maximum(torch.minimum(p1*p2*(p3-T), S/dt), torch.tensor(0));
    def melt_1(self,p1,p2,T,S,dt):
        return torch.maximum(torch.minimum(p1*(T-p2),S/dt),torch.tensor(0));
    def rainfall_2(self,In,T,p1,p2):
        return torch.minimum(In,torch.maximum(torch.tensor(0),In*(T-(p1-0.5*p2))/p2));
    def infiltration_3(self, In,S,Smax):
        # mask = S>Smax
        # return In*mask
        return In*(torch.tensor(1)-self.smoothThreshold_storage_logistic(S,Smax));
    def smoothThreshold_storage_logistic(self,S,Smax):
        r = torch.tensor(0.01);
        e = 5.00;

        #% Calculate multiplier
        Smax = torch.maximum(Smax,torch.tensor(0));  # % this avoids numerical instabilities when Smax<0

        out = 1 / (1+torch.exp((S-Smax+r*e*Smax)/torch.maximum(r,r*Smax)));


        return out
    def excess_1(self,So,Smax,dt):
        return torch.maximum((So-Smax)/dt,torch.tensor(0));
    def capillary_1(self,p1,S1,S1max,S2,dt):
        return  torch.minimum(p1*(torch.tensor(1)-S1/S1max),S2/dt);


    def evap_3(self,p1,S,Smax,Ep,dt):

        return torch.min(torch.cat((S/(p1*Smax)*Ep.unsqueeze(0),Ep.unsqueeze(0),S/dt.unsqueeze(0).unsqueeze(0)), 0),0).values;
    def recharge_2(self,p1,S,Smax,flux):
        return flux*((torch.maximum(S,torch.tensor(0))/Smax)**p1);
    def interflow_2(self,p1,S,p2,dt):
        return torch.minimum(p1*torch.maximum(S,torch.tensor(0))**(1+p2),torch.maximum(S/dt,torch.tensor(0)));
    def percolation_1(self,p1,S,dt):
        return torch.minimum(p1,S/dt);

    def baseflow_1(self,p1,S):
        return p1*S;
