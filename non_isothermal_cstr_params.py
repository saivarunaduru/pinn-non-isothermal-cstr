from dataclasses import dataclass

@dataclass
class cstr_params:
    k0:float    = 10**10
    rho:float   = 10**6
    Cp:float    = 1.0
    deltaHr:float = -130*10**6
    rhoc:float  = 10**6
    Cpc:float   = 1.0
    V:float     = 1.0
    Tcin:float  = 365
    T0:float    = 400
    a:float     = 1.678*10**6
    E_by_R:float= 8330
    b:float     = 0.5

    Fc:float    = 15.0
    F:float     = 1.0
    X1:float    = 0.265
    X2:float    = 393.954
    CA0:float   = 0.25 #2

    def __post_init__(self):
        self.UA = (self.a *((self.Fc)**(self.b+1))/ \
                 (self.Fc + ((self.a*self.Fc)**self.b)/(2*self.rhoc*self.Cpc)))
        self.scales = [self.CA0, self.T0]


@dataclass
class training_params:
    
    num_outputs: int = 2
    num_layers: int = 4
    num_neurons: int = 64
       
    lambda_ic:float=75
    lr:float=1e-3
    num_samples:int = 4000
    t_min_ratio:float = 1e-3

    t_scale:float = None
    