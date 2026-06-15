import torch
import torch.nn as nn
from nonlinearSolver.NewRaphson import NRODESolver
class NRBacksolveFunction(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(
        self,
        y0,
        theta,
        delta_t,
        input,
        nflux,
        g

    ):

        bs,ny = y0.shape
        rho = input.shape[0]



        NR = NRODESolver(y0, g, delta_t)
        ySolution = torch.zeros((rho,bs,ny)).to(y0)

        fluxSolution = torch.zeros((rho,bs,nflux)).to(y0)

        Residual = torch.zeros((rho,bs,ny)).to(y0)
        NRFlag = torch.zeros((rho)).to(y0)
        ySolution[0,:,:] = y0

        for day in range(rho):
            if day == 0:
                yold = ySolution[0,:,:]
            else:
                yold = ySolution[day-1,:,:]
            t = torch.tensor(day).to(yold)
            yNew, F, exitflag = NR(yold,delta_t,t)

            dy, flux = g(t,yNew);

            ySolution[day,:,:]  = yold + dy*delta_t
            Residual[day,:,:]  =  F
            fluxSolution[day,:,:]  =  flux*delta_t
            NRFlag[day]  =   exitflag



        return ySolution, fluxSolution, Residual,NRFlag
