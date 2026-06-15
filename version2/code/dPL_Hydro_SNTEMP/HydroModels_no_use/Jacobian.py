import torch
from functorch import vmap
def batchJacobian_AD_new(x,y,graphed=False,doSqueeze=True):
    # extract the jacobian dy/dx for multi-column y output (and with minibatch)
    # compared to the scalar version above, this version will call grad() ny times and store outputs in a tensor matrix
    # y: [nb, ny]; x: [nb, nx]. x could also be a tuple or list of tensors.
    # permute and view your y to be of the above format.
    # AD jacobian is not free and may end up costing me time
    # output: Jacobian [nb, ny, nx] # will squeeze after the calculation
    # relying on the fact that the minibatch has nothing to do with each other!
    # if they do, i.e, they come from different time steps of a simulation, you need to put them in second dim in y!
    # view or reshape your x and y to be in this format if they are not!
    # pay attention, this operation could be expensive.
    if y.ndim==1: # could've called batchScalarJacobian_AD() but we can handle this anyway
        y = y.unsqueeze(1)
    ny = y.shape[-1]
    b = x.shape[0]
    v = torch.zeros([b,ny,ny]).to(y)
    # to get rid of the nan values where y == 0.0
    # y = torch.where(y == 0.0,
    #                 y + 0.000001,
    #                 y)
    # Farshid's comment:'
    # we can the following one line code for the for loop below:
    # v = torch.eye(ny, ny).repeat(b, 1, 1).to(y)
    for i in range(ny):
      v[:,i,i]=1
    def get_vjp1(v):
        return torch.autograd.grad(outputs=y, inputs=x, grad_outputs=v,retain_graph=True,create_graph=graphed)

    DYDX = vmap(get_vjp1,in_dims=(1),out_dims=1)(v)[0]
    if not graphed:
        # during test, we may detach the graph
        # without doing this, the following cannot be cleaned from memory between time steps as something use them outside
        # however, if you are using the gradient during test, then graphed should be false.
        DYDX = DYDX.detach()
        x = x.detach()
    return DYDX
