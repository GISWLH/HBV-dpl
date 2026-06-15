import torch
import torch.nn as nn

class HyMod(nn.Module):
    def __init__(self, theta,delta_t,climate_data):
        super().__init__()

        self.register_parameter("theta", theta)
        self.delta_t = delta_t
        self.climate_data = climate_data

    def forward(self, t, y):
        ##parameters
        smax   = self.theta[0];    # % Maximum soil moisture storage     [mm],
        b      = self.theta[1];    # % Soil depth distribution parameter [-]
        a      = self.theta[2];    # % Runoff distribution fraction [-]
        kf     = self.theta[3];    # % Fast runoff coefficient [d-1]
        ks     = self.theta[4];   #  % Slow runoff coefficient [d-1]

        ##% stores
        S1 = y[:,0].clone();
        S2 = y[:,1].clone();
        S3 = y[:,2].clone();
        S4 = y[:,3].clone();
        S5 = y[:,4].clone();
        dS = torch.zeros(y.shape).to(y)
        fluxes = torch.zeros((y.shape[0],8)).to(y)

        climate_in = self.climate_data[int(t),:,:];   ##% climate at this step
        P  = climate_in[:,0];
        Ep = climate_in[:,1];
        T  = climate_in[:,2];

        ##% fluxes functions
        flux_ea  = self.evap_7(S1,smax,Ep,self.delta_t);
        flux_pe  = self.saturation_2(S1,smax,b,P);
        flux_pf  = self.split_1(a,flux_pe);
        flux_ps  = self.split_1(1-a,flux_pe);
        flux_qf1 = self.baseflow_1(kf,S2);
        flux_qf2 = self.baseflow_1(kf,S3);
        flux_qf3 = self.baseflow_1(kf,S4);
        flux_qs  = self.baseflow_1(ks,S5);

        ##% stores ODEs
        dS[:,0] = P - flux_ea - flux_pe;
        dS[:,1] = flux_pf - flux_qf1;
        dS[:,2] = flux_qf1 - flux_qf2;
        dS[:,3] = flux_qf2 - flux_qf3;
        dS[:,4] = flux_ps - flux_qs;
        fluxes[:,0] =flux_ea
        fluxes[:,1] =flux_pe
        fluxes[:,2] =flux_pf
        fluxes[:,3] =flux_ps
        fluxes[:,4] =flux_qf1
        fluxes[:,5] =flux_qf2
        fluxes[:,6] =flux_qf3
        fluxes[:,7] =flux_qs

        return dS,fluxes

    def evap_7(self,S,Smax,Ep,dt):
        return torch.minimum(S/Smax*Ep,S/dt);

    def saturation_2(self,S,Smax,p1,In):
        return (1- torch.minimum(torch.tensor(1.0),torch.maximum(torch.tensor(0.0),(1.0-S/Smax)))**p1)*In;

    def split_1(self,p1,In):
        return p1*In;

    def baseflow_1(self,p1,S):
        return p1*S;
