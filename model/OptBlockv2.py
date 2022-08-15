import numpy as np
import tensorflow as tf


class OptBlockv2(tf.keras.Model):
    def __init__(self, payload, mip_solver, from_mip_sol=True):
        super(OptBlockv2, self).__init__(name="")

        self.payload = payload
        self.mip_solver = mip_solver

        self.create_tf_vars(from_mip_sol=from_mip_sol)

    def create_tf_vars(self, from_mip_sol=True):
        self.tf_cur_inven = {}
        self.mip_upd_inven = {}
        self.tf_tar_alloc = {}
        self.mip_target = []

        for shop in self.payload.shops:
            self.tf_cur_inven[shop] = {}
            self.mip_upd_inven[shop] = {}
            self.tf_tar_alloc[shop] = {}
            for style in self.payload.styles:
                for color in self.payload.colors:
                    sc_units = self.payload.current_inventory[shop][style + color]
                    self.tf_cur_inven[shop][style + color] = tf.cast(
                        self.payload.current_inventory[shop][style + color], tf.float32
                    )
                    self.tf_tar_alloc[shop][style + color] = tf.cast(
                        self.payload.target_allocation[shop][style + color], tf.float32
                    )
                    if self.mip_solver is not None:
                        for bundle in self.payload.bundles:
                            sc_units += (
                                self.mip_solver.var_inventory_alloc[shop, bundle]
                                * self.payload.bundles_sc[bundle][style + color]
                            )

                        self.mip_upd_inven[shop][style + color] = tf.cast(
                            sc_units, tf.float32
                        )
                        self.mip_target.append(
                            [self.mip_upd_inven[shop][style + color]]
                        )
                    else:
                        self.mip_target.append([self.tf_tar_alloc[shop][style + color]])

        self.mip_target = tf.reshape(tf.concat(self.mip_target, axis=0), [1, -1])

        self.tf_proba = {}
        self.mip_shop_bundle = {}
        self.tf_alloc_shp_bun = {}
        i = 0
        for shop in self.payload.shops:
            for bundle in self.payload.bundles:
                self.tf_proba[(shop, bundle)] = tf.cast(0.0, tf.float32)
                self.tf_alloc_shp_bun[(shop, bundle)] = tf.cast(0.0, tf.float32)
                if self.mip_solver is not None:
                    self.mip_shop_bundle[(shop, bundle)] = tf.cast(
                        self.mip_solver.var_inventory_alloc[shop, bundle], tf.float32
                    )

        self.mip_alloc_bundle = {}
        self.tf_alloc_bundle = {}
        self.tf_avail_bundle = {}

        for bundle in self.payload.bundles:
            if self.mip_solver is not None:
                self.mip_alloc_bundle[bundle] = tf.cast(
                    self.mip_solver.var_bundle_gather[bundle], tf.float32
                )
            self.tf_alloc_bundle[bundle] = tf.cast(0.0, tf.float32)
            self.tf_avail_bundle[bundle] = tf.cast(
                self.payload.bundle_availability[bundle], tf.float32
            )

        self.tf_bun_sty_col = {}
        for bundle in self.payload.bundles:
            self.tf_bun_sty_col[bundle] = {}
            for style in self.payload.styles:
                for color in self.payload.colors:
                    self.tf_bun_sty_col[bundle][style + color] = tf.cast(
                        self.payload.bundles_sc[bundle][style + color], tf.float32
                    )

    def call(self, proba, training=False):

        next_inven = {}
        for shop in self.payload.shops:
            next_inven[shop] = {}
            for style in self.payload.styles:
                for color in self.payload.colors:
                    next_inven[shop][style + color] = self.tf_cur_inven[shop][
                        style + color
                    ]
       

        i = 0
        for shop in self.payload.shops:
            for bundle in self.payload.bundles:
                for style in self.payload.styles:
                    for color in self.payload.colors:

                        constraint_multiplier = 1.0
                        if (self.tf_avail_bundle[bundle] - proba[i] < -0.0) and proba[i]>0:
                            constraint_multiplier = 0.01

                        if (self.tf_alloc_bundle[bundle] + proba[i] < -0.0) and proba[i]<0:
                            constraint_multiplier = 0.01  
                            
                        next_inven[shop][style + color] += (
                            self.tf_bun_sty_col[bundle][style + color] * proba[i] * constraint_multiplier
                        )
                i += 1

        result = []
        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    result.append([next_inven[shop][style + color]])

        result = tf.reshape(tf.concat(result, axis=0), [1, -1])

        if training == True:

            for shop in self.payload.shops:
                for style in self.payload.styles:
                    for color in self.payload.colors:

                        self.tf_cur_inven[shop][style + color] = next_inven[shop][
                            style + color
                        ]
            i = 0
            for shop in self.payload.shops:
                for bundle in self.payload.bundles:

                    constraint_multiplier = 1.0
                    if (self.tf_avail_bundle[bundle] - proba[i] < -0.0) and proba[i]>0:
                        constraint_multiplier = 0.01

                    if (self.tf_alloc_bundle[bundle] + proba[i] < -0.0) and proba[i]<0:
                        constraint_multiplier = 0.01     

                    self.tf_avail_bundle[bundle] -= constraint_multiplier * proba[i]
                    self.tf_alloc_bundle[bundle] += constraint_multiplier * proba[i]
                    self.tf_alloc_shp_bun[(shop, bundle)] += constraint_multiplier * proba[i]
                    self.tf_proba[(shop, bundle)] = constraint_multiplier * proba[i]
                    # else:
                    #     print(f'did not update {shop}, {bundle}, negative bundle availability')
                    i += 1

        return result

    def get_bun_sku_lkup(self):
        bun_lkup = []
        for bundle in self.payload.bundles:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    bun_lkup.append([self.tf_bun_sty_col[bundle][style + color]])
        bun_lkup = tf.reshape(tf.concat(bun_lkup, axis=0), [1, -1])
        return bun_lkup

    def set_bun_sku_lkup(self, x):
        x = tf.squeeze(x)
        i = 0
        for bundle in self.payload.bundles:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    self.tf_bun_sty_col[bundle][style + color] = x[i]
                    i += 1
        return

    def get_ava_bun(self):
        ava_bun = []
        for bundle in self.payload.bundles:
            ava_bun.append([self.tf_avail_bundle[bundle]])
        ava_bun = tf.reshape(tf.concat(ava_bun, axis=0), [1, -1])
        return ava_bun

    def set_ava_bun(self, x):
        x = tf.squeeze(x)
        i = 0
        for bundle in self.payload.bundles:
            self.tf_avail_bundle[bundle] = x[i]
            i += 1
        return

    def get_cur_inv(self):
        cur_inventory = []
        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    cur_inventory.append([self.tf_cur_inven[shop][style + color]])
        cur_inventory = tf.reshape(tf.concat(cur_inventory, axis=0), [1, -1])
        return cur_inventory

    def set_cur_inv(self, x):
        x = tf.squeeze(x)
        i = 0
        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    self.tf_cur_inven[shop][style + color] = x[i]
                    i += 1
        return

    def get_tar_alloc(self):
        tar_inventory = []
        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:
                    tar_inventory.append([self.tf_tar_alloc[shop][style + color]])
        tar_inventory = tf.reshape(tf.concat(tar_inventory, axis=0), [1, -1])
        return tar_inventory

    def get_shop_bun_alloc(self):

        shop_bundle_alloc = []
        for shop in self.payload.shops:
            for bundle in self.payload.bundles:
                shop_bundle_alloc.append([self.tf_alloc_shp_bun[(shop, bundle)]])

        shop_bundle_alloc = tf.reshape(tf.concat(shop_bundle_alloc, axis=0), [1, -1])
        return shop_bundle_alloc

    def set_shop_bun_alloc(self, x):
        x = tf.squeeze(x)
        i = 0
        for shop in self.payload.shops:
            for bundle in self.payload.bundles:
                self.tf_alloc_shp_bun[(shop, bundle)] = x[i]
                i += 1
        return

    def get_shop_bun_proba(self):

        shop_proba = []
        for shop in self.payload.shops:
            for bundle in self.payload.bundles:
                shop_proba.append([self.tf_proba[(shop, bundle)]])

        shop_proba = tf.reshape(tf.concat(shop_proba, axis=0), [1, -1])
        return shop_proba

    def set_shop_bun_proba(self, x):
        x = tf.squeeze(x)
        i = 0

        for shop in self.payload.shops:
            for bundle in self.payload.bundles:
                self.tf_proba[(shop, bundle)] = x[i]
                i += 1
        return

    def get_bun_ava_sku_map(self):

        bun_map = []

        for style in self.payload.styles:
            for color in self.payload.colors:
                sc_units = 0
                for bundle in self.payload.bundles:
                    sc_units += (
                        tf.cast(self.payload.bundle_availability[bundle], tf.float32)
                        * self.payload.bundles_sc[bundle][style + color]
                    )

                bun_map.append([sc_units])

        bun_map = tf.reshape(tf.concat(bun_map, axis=0), [1, -1])
        return bun_map

    # Get the percent of shop-SKU allocations for negative available bundles
    # tf_avail_bundle is updated when the Block is called with training=True
    def get_neg_alloc_sku_map(self):

        bun_map = []

        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:

                    sc_units = 0.0
                    for bundle in self.payload.bundles:

                        if self.tf_alloc_shp_bun[(shop, bundle)] <= 0.0:

                            sc_units += (
                                self.tf_alloc_shp_bun[(shop, bundle)]
                                * self.payload.bundles_sc[bundle][style + color]
                            )       


                    bun_map.append([sc_units])
                    # number of bundles in a shop

        bun_map = tf.reshape(tf.concat(bun_map, axis=0), [1, -1])
        return bun_map

    def get_over_alloc_sku_map(self):

        bun_map = []

        for shop in self.payload.shops:
            for style in self.payload.styles:
                for color in self.payload.colors:

                    sc_units = 0.0
                    for bundle in self.payload.bundles:

                        # There is a negative bundle allocation, how does that effect this accumulator?
                        if self.tf_avail_bundle[bundle] <= -0.0 and self.tf_alloc_shp_bun[(shop, bundle)] >= 0:

                            sc_units += (
                                self.tf_alloc_shp_bun[(shop, bundle)]
                                * self.payload.bundles_sc[bundle][style + color]
                                * -1 * (self.tf_avail_bundle[bundle]/max(1,self.payload.bundle_availability[bundle]))
                            )        


                    bun_map.append([sc_units])
                    # number of bundles in a shop

        bun_map = tf.reshape(tf.concat(bun_map, axis=0), [1, -1])
        return bun_map

    # def get_mip_shop_bundle(self):
    #     tar_shp_bun = []
    #     for shop in self.payload.shops:
    #         for bundle in self.payload.bundles:
    #          tar_shp_bun.append([self.mip_shop_bundle[(shop,bundle)]])

    #     tar_shp_bun = tf.reshape(tf.concat(tar_shp_bun,axis=0),[1,-1])
    #     return tar_shp_bun
