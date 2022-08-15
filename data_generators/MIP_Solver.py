import numpy as np
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Optional, TypeVar
import uuid
import pickle 
from . import Payload
import pulp

@dataclass
class MIP_Solver():
    
    """Create an Mip Solver object to store MIP solution data
    """

    payload: TypeVar('payload')
        
    def __post_init__(self):
        
        self._p_b_var = pulp.LpVariable.dicts('bundle_gather', self.payload.bundles,lowBound=0, cat="Integer")
        self._p_ba_var = pulp.LpVariable.dicts('bundle_alloc', self.payload.bundle_allocation,lowBound=0, cat="Integer")
        self._p_id_var = pulp.LpVariable.dicts('inventory_diff', self.payload.inventory_diff, cat="Binary")
        self._p_ia_var = pulp.LpVariable.dicts('inventory_alloc', self.payload.inventory_allocation,lowBound=0, cat="Integer")

        model = pulp.LpProblem("opt_inventory", pulp.LpMaximize)

        cost_obj = pulp.lpSum(self._p_b_var)

        cost_obj

        model+=cost_obj
        
        for bundle in self.payload.bundles:
            model+=self.payload.bundle_availability[bundle]-self._p_b_var[bundle] >= 0


        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    constr=[]
                    for i in range(self.payload.loss_bias_lb,self.payload.loss_bias_ub+1):
                        constr.append(self._p_id_var[(shop,style+color,i)])
                    model += pulp.lpSum(constr)==1

        for bundle in self.payload.bundles_sc:

            for style in self.payload.styles:
                for color in self.payload.colors:
                    #constraint on bundle and allocation
                    model += (self._p_b_var[bundle] * self.payload.bundles_sc[bundle][style+color] 
                                  == self._p_ba_var[bundle,style+color]
                             )
            ia_constr = []        
            for shop in self.payload.shops:         
                ia_constr.append(self._p_ia_var[shop,bundle])
            #make sure that the all distributor bundles are sent to shops
            model += self._p_b_var[bundle]==pulp.lpSum(ia_constr)


        #efficiency of allocation
        allocation_eff = []
        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    sc_units = self.payload.current_inventory[shop][style+color]
                    for bundle in self.payload.bundles:
                        sc_units += self._p_ia_var[(shop,bundle)]*self.payload.bundles_sc[bundle][style+color] 

                    #allocated bundles*number of shirts in bundle_id for each style color
                    loss = sc_units - self.payload.target_allocation[shop][style+color]
                    #ensure it does not go over the desired allocation
                    model+= loss <= self.payload.loss_bias_ub
                    model+= loss >= self.payload.loss_bias_lb

                    allocation_eff.append(loss)


                    id_var_constr = []
                    for i in range(self.payload.loss_bias_lb,self.payload.loss_bias_ub+1):
                        id_var_constr.append(self._p_id_var[(shop,style+color,i)]*i)

                    model+=pulp.lpSum(id_var_constr)==loss

        non_lin = [np.sqrt(i**2) for i in range(self.payload.loss_bias_lb,self.payload.loss_bias_ub+1)]


        custom_alloc_eff = []
        i=0
        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    new_loss = 0
                    for j in range(self.payload.loss_bias_lb,self.payload.loss_bias_ub+1):
                        new_loss+=self._p_id_var[(shop,style+color,j)]*non_lin[j]
                    custom_alloc_eff.append(new_loss)
                    i+=1

        allocation_obj = pulp.lpSum(custom_alloc_eff)

        model+=allocation_obj

        solution = model.solve()

        print("The Objective Value", pulp.value(model.objective))

        self._p_b_var = self.convert_var_to_dict(self._p_b_var)
        self._p_ba_var = self.convert_var_to_dict(self._p_ba_var)
        self._p_id_var = self.convert_var_to_dict(self._p_id_var)
        self._p_ia_var = self.convert_var_to_dict(self._p_ia_var)

    def convert_var_to_dict(self, var):
        return {key:var[key].value() for key in var.keys()}
        
        
    @property
    def var_bundle_gather(self):
        return self._p_b_var
    

    @property
    def var_bundle_alloc(self):
        return self._p_ba_var
    
    @property
    def var_inventory_diff(self):
        return self._p_id_var
    
    @property
    def var_inventory_alloc(self):
        return self._p_ia_var
