import numpy as np
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional, TypeVar
import uuid
import pickle 

@dataclass
class Payload():
    
    """Creates a Payload object to store randomly generated data
    """

    # make sure there is a +1 to the uppber bound (ub) in range functions

    seed: int
    uniq_id: str
    dataset: str
    bundle_noise: int = 1
    
    bundle_avail_lb: int = 5
    bundle_avail_ub: int = 20
        
    current_inv_lb: int = 0
    current_inv_ub: int = 10
        

    target_inv_noise_lb: int = 0
    target_inv_noise_ub: int = 60
        
    loss_bias_lb: int = -50
    loss_bias_ub: int = 50


    def __post_init__(self):
        
            
        assert self.bundle_avail_lb < self.bundle_avail_ub
        assert self.bundle_avail_lb >=0 

        assert self.current_inv_lb < self.current_inv_ub
        assert self.current_inv_lb >= 0
        
        assert self.target_inv_noise_lb < self.target_inv_noise_ub
            
        self.loss_bias_ub = -self.target_inv_noise_ub
        self.loss_bias_ub = self.target_inv_noise_ub

        assert self.loss_bias_lb < self.loss_bias_ub


        np.random.seed(self.seed)
        
        N_bundles = 13
        
        self._styles = ["A","B","C"]
        self._colors= ["R", "B", "G"]
        self._shops = ["1","2","3"]
        self._bundles = [str(i+1) for i in range(N_bundles)]
        
        self._bundles_sc={
            "1":{"AR":1,"AB":1,"AG":1,"BR":1,"BB":1,"BG":1,"CR":1,"CB":1,"CG":1},
            "2":{"AR":3,"AB":3,"AG":3,"BR":0,"BB":0,"BG":0,"CR":0,"CB":0,"CG":0},
            "3":{"AR":0,"AB":0,"AG":0,"BR":3,"BB":3,"BG":3,"CR":0,"CB":0,"CG":0},
            "4":{"AR":0,"AB":0,"AG":0,"BR":0,"BB":0,"BG":0,"CR":3,"CB":3,"CG":3},
            "5":{"AR":1,"AB":0,"AG":0,"BR":0,"BB":0,"BG":0,"CR":0,"CB":0,"CG":0},
            "6":{"AR":0,"AB":1,"AG":0,"BR":0,"BB":0,"BG":0,"CR":0,"CB":0,"CG":0},
            "7":{"AR":0,"AB":0,"AG":1,"BR":0,"BB":0,"BG":0,"CR":0,"CB":0,"CG":0},
            "8":{"AR":0,"AB":0,"AG":0,"BR":1,"BB":0,"BG":0,"CR":0,"CB":0,"CG":0},
            "9":{"AR":0,"AB":0,"AG":0,"BR":0,"BB":1,"BG":0,"CR":0,"CB":0,"CG":0},
            "10":{"AR":0,"AB":0,"AG":0,"BR":0,"BB":0,"BG":1,"CR":0,"CB":0,"CG":0},
            "11":{"AR":0,"AB":0,"AG":0,"BR":0,"BB":0,"BG":0,"CR":1,"CB":0,"CG":0},
            "12":{"AR":0,"AB":0,"AG":0,"BR":0,"BB":0,"BG":0,"CR":0,"CB":1,"CG":0},
            "13":{"AR":0,"AB":0,"AG":0,"BR":0,"BB":0,"BG":0,"CR":0,"CB":0,"CG":1}
        }

        # shuffle bundle ordering
        # bundle_keys = list(self._bundles_sc.keys())
        # np.random.shuffle(bundle_keys)
        # self._bundles_sc = {str(i+1):self._bundles_sc[key] for i,key in enumerate(bundle_keys)}
        
                    
        for i in range(N_bundles):
            for style in self._styles:
                for color in self._colors:
                    self._bundles_sc[str(i+1)][style+color] = max(0,
                                                                  self._bundles_sc[str(i+1)][style+color] + 
                                                                  np.random.randint(-self.bundle_noise,self.bundle_noise+1)
                                                                 )
        

        self._bundle_availability = {str(i+1):np.random.randint(self.bundle_avail_lb,self.bundle_avail_ub+1)
                                        for i in range(N_bundles)}

        self._inventory_diff = []
        self._current_inventory = {}
        self._target_allocation = {}
        for i,shop in enumerate(self.shops):
            self._current_inventory[shop] = {}
            self._target_allocation[shop] = {}
            for style in self._styles:
                for color in self._colors:
                    self._current_inventory[shop][style+color] = np.random.randint(self.current_inv_lb,self.current_inv_ub+1)
                    self._target_allocation[shop][style+color] = max(self.current_inventory[shop][style+color],
                                                                    self.current_inventory[shop][style+color] + 
                                                                    np.random.randint(self.target_inv_noise_lb,
                                                                                      self.target_inv_noise_ub+1)
                                                                      )

                    
                    for i in range(self.loss_bias_lb,self.loss_bias_ub+1):
                        self._inventory_diff.append((shop,style+color,i))
                        

        self._bundle_allocation = []
        #bundle k1
        for k1 in self._bundles_sc.keys():
            #shop+color k2
            for k2 in self._bundles_sc[k1].keys():
                self._bundle_allocation.append((k1,k2))

        self._inventory_allocation = []
        for shop in self._shops:
            for bundle in self._bundles:
                self._inventory_allocation.append((shop,bundle))
                        

    @property
    def styles(self):
        return self._styles
    
    @property
    def colors(self):
        return self._colors
    

    @property
    def shops(self):
        return self._shops
    

    @property
    def bundles(self):
        return self._bundles
    
    @property
    def bundles_sc(self):
        return self._bundles_sc
    
    
    @property
    def bundle_availability(self):
        return self._bundle_availability
        
    @property
    def current_inventory(self):
        return self._current_inventory
    
    @property
    def target_allocation(self):
        return self._target_allocation
    
     
    @property
    def inventory_diff(self):
        return self._inventory_diff
    
    
    @property
    def bundle_allocation(self):
        return self._bundle_allocation
    
    @property
    def inventory_allocation(self):
        return self._inventory_allocation