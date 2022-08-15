import tensorflow as tf
from . import OptBlockv2
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], "GPU")
visible_devices = tf.config.get_visible_devices()


@tf.custom_gradient
def equality_gradient(c, l, t, tol = 5e-2):
    def grad(ys):
        
        if l<=0:
            h = tf.math.abs(c)*1e2
        else:
            h = c*0.1
        

        diff = tf.math.abs(t-c)
        if diff>tol and c>t:
            g = tf.math.abs(l)*1e2
        elif diff>tol and c<t:
            g = -tf.math.abs(l)*1e2
        else:
            g = l
            
        return g, h, None

    return tf.math.abs(l)*c , grad


@tf.function
def sumOfSquareErrors(ytrue, ypred):

    loss = tf.reduce_sum(tf.square(ypred - ytrue))
    return loss


@tf.function
def rewardFunction(ytrue, ypred, sku_limit, neg_bun_alloc, over_bun_alloc):

    # Create booleans
    is_neg_alloc = tf.cast(ypred < 0.0, tf.float32)
    # is_neg_bund_alloc = tf.cast(neg_bun_alloc < 0.0, tf.float32)
    # is_over_bund_alloc = tf.cast(over_bun_alloc > 0.0, tf.float32)

    # Remove negative allocations as those are errors
    # sum_across_sku = tf.reduce_sum(ypred * (1 - is_neg_alloc), axis=0)

    # If the allocated sum of exploded skus is greater than what is available, create a boolean
    # sku_limit = tf.reshape(tf.cast(sum_across_sku - sku_limit > 0, tf.float32), [1, -1])

    # where ytrue > ypred, otherwise zero
    pos_reward = tf.clip_by_value(ytrue - ypred, 0, 1e10)

    neg_reward = tf.clip_by_value(ytrue - ypred, -1e10, 0)
   
    is_over_bun = tf.cast(over_bun_alloc > 0, tf.float32)
    is_neg_bun = tf.cast(neg_bun_alloc < 0, tf.float32)

    reward = (
        pos_reward * (1 - 0.99*is_over_bun)
        + neg_reward * (1 - 0.99*is_neg_bun)
        + 0.10 * is_neg_alloc * pos_reward
        # - 0.20 * sku_limit * (0.5 * pos_reward - neg_reward)
        # If negative bundle limit is hit, add 10% of the positive reward and remove negative reward by 20%
        - 1.01 * neg_bun_alloc
        # If the bundle limit is hit, remove 10% of the positive reward and add negative reward by 20%
        - 1.01 * over_bun_alloc
    )

    return tf.reshape(reward, [1, -1])


class OptModelv2:
    def __init__(self, alpha=0.99, mip_sol_wgt=0.99):

        self.loop_cnt = 0
        self.vid_count = 0
        self.gamma = 1e-2
        self.alpha = alpha
        self.mip_sol_wgt = mip_sol_wgt

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.loss_function = sumOfSquareErrors

        constraint_var_init = tf.random.uniform(shape=(1, 1),minval=0.1, maxval=1)
        self.constraint_var = tf.Variable(constraint_var_init, name="constraint_var",trainable=True,dtype=tf.float32)

        self.prev_loss = 1e6

    def init_block(self, payload, mip_solver):
        self.prev_loss = 1e6
        self.block = OptBlockv2(payload, mip_solver)
        if self.loop_cnt == 0:
            print("Create new model")
            self.model = self.create_model()

    def create_model(self):

        bun_lkup = self.block.get_bun_sku_lkup()
        ava_bun = self.block.get_ava_bun()
        cur_inventory = self.block.get_cur_inv()
        neg_alloc_sku_map = self.block.get_neg_alloc_sku_map()
        over_alloc_sku_map = self.block.get_over_alloc_sku_map()
        shop_bundle_alloc = self.block.get_shop_bun_alloc()
        shop_bun_proba = self.block.get_shop_bun_proba()
        tar_alloc = self.block.get_tar_alloc()

        in_bun_lkup = tf.keras.Input(shape=bun_lkup.shape[1])
        in_ava_bund = tf.keras.Input(shape=ava_bun.shape[1])
        in_cur_inven = tf.keras.Input(shape=cur_inventory.shape[1])
        in_neg_alloc_sku_map = tf.keras.Input(shape=neg_alloc_sku_map.shape[1])
        in_over_alloc_sku_map = tf.keras.Input(shape=over_alloc_sku_map.shape[1])
        in_shop_bun = tf.keras.Input(shape=shop_bundle_alloc.shape[1])
        in_shop_bun_prob = tf.keras.Input(shape=shop_bun_proba.shape[1])
        in_tar_alloc = tf.keras.Input(shape=tar_alloc.shape[1])

        N_shops = len(self.block.payload.shops)
        N_bundles = len(self.block.payload.bundles)
        N_style_clors = len(self.block.payload.styles) * len(self.block.payload.colors)

        remain_sc = tf.reshape(in_bun_lkup, [1, N_bundles, N_style_clors])
        remain_sc = tf.transpose(remain_sc, perm=[0, 2, 1])
        remain_sc = remain_sc * in_ava_bund
        remain_sc = tf.keras.layers.Flatten()(remain_sc)


        is_neg_alloc = tf.cast(in_cur_inven < 0.0, tf.float32)
        is_over_bun = tf.cast(in_over_alloc_sku_map > 0, tf.float32)
        is_neg_bun = tf.cast(in_neg_alloc_sku_map < 0, tf.float32)


        # available bundles, allocated shop_bundles, prev_shop_bund
        # allocated shop_style_colors, remaining style_colors
        # available style_colors

        out_shape = shop_bun_proba.shape[1]

        x = tf.keras.layers.Concatenate()(
            [   
                in_shop_bun,
                in_shop_bun_prob,
                in_ava_bund,
                in_shop_bun + in_shop_bun_prob,
                in_shop_bun - in_shop_bun_prob,
                remain_sc,
                in_tar_alloc,
                in_cur_inven,
                in_neg_alloc_sku_map,
                in_over_alloc_sku_map,
                is_neg_alloc,
                is_over_bun,
                is_neg_bun,
                in_tar_alloc - in_cur_inven,
                in_tar_alloc - in_cur_inven - 1.01*in_neg_alloc_sku_map - 1.01*in_over_alloc_sku_map
            ]
        )
        
        x = tf.keras.layers.Dense(N_shops*N_shops*N_bundles*50, activation="linear",use_bias=True)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(N_shops*N_shops*N_bundles*20, activation="linear",use_bias=True)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(N_shops*N_shops*N_bundles*20, activation="linear",use_bias=True)(x)
        x = tf.keras.layers.Dropout(0.1)(x)   
        x = tf.keras.layers.Dense(N_shops*N_shops*N_bundles*20, activation="linear",use_bias=True)(x)
        x = tf.keras.layers.Dropout(0.1)(x)   
        x = tf.keras.layers.Dense(N_shops*N_shops*N_bundles*10, activation="linear",use_bias=False)(x)
        x = tf.keras.layers.Dense(N_shops*N_shops*N_bundles*10, activation="linear",use_bias=False)(x)

        
        scale = 0.001
        shift = 0.0
        output = (
            scale
            * tf.keras.layers.Dense(out_shape, activation="linear", use_bias=False)(x)
            + shift
        )

        inputs = [
            in_bun_lkup,
            in_ava_bund,
            in_cur_inven,
            in_neg_alloc_sku_map,
            in_over_alloc_sku_map,
            in_shop_bun,
            in_shop_bun_prob,
            in_tar_alloc
        ]

        model = tf.keras.Model(inputs, output)
        return model

    def batch_train(self, move_num=0):
        # get the current state

        self.loop_cnt += 1

        z_ava_bun_backup = self.block.get_ava_bun()
        z_cur_inventory_backup = self.block.get_cur_inv()
        z_shop_bundle_alloc_backup = self.block.get_shop_bun_alloc()
        z_proba_backup = self.block.get_shop_bun_proba()

        z_bun_lkup = self.block.get_bun_sku_lkup()
        z_bun_ava_sku_map = self.block.get_bun_ava_sku_map()
        z_tar_alloc = self.block.get_tar_alloc()

        #         if tf.reduce_sum(tf.cast(z_ava_bun_backup<-3,tf.float32)):
        #             print('stopping, no available bundles')
        #             return 0

        #         if tf.reduce_sum(tf.cast(z_cur_inventory_backup<-3,tf.float32)):
        #             print('stopping, negative inventory')
        #             return 0

        N_moves_left = 10

        for i in range(N_moves_left):

            z_ava_bun = self.block.get_ava_bun()
            z_cur_inventory = self.block.get_cur_inv()
            z_shop_bundle_alloc = self.block.get_shop_bun_alloc()
            z_proba = self.block.get_shop_bun_proba()

            # this is derived from z_ava_bun
            z_neg_alloc_sku_map = self.block.get_neg_alloc_sku_map()
            z_over_alloc_sku_map = self.block.get_over_alloc_sku_map()

            # tranche
            # Caculate the next state without training model
            rl_update = self.model(
                [
                    tf.cast(z_bun_lkup, tf.float32),
                    tf.cast(z_ava_bun, tf.float32),
                    tf.cast(z_cur_inventory, tf.float32),
                    tf.cast(z_neg_alloc_sku_map, tf.float32),
                    tf.cast(z_over_alloc_sku_map, tf.float32),
                    tf.cast(z_shop_bundle_alloc, tf.float32),
                    tf.cast(z_proba, tf.float32),
                    tf.cast(z_tar_alloc, tf.float32),
                ],
                training=False,
            )

            rl_update = tf.squeeze(rl_update)

            #             proba_update=tf.clip_by_value(tf.squeeze(proba_update),-1.,1.)
            #             proba_update=tf.squeeze(proba_update)
            #             print(proba_update.numpy())

            q_proba = tf.squeeze(z_proba) + self.gamma * rl_update

            #             q_proba=tf.clip_by_value(tf.squeeze(z_proba)+self.gamma*proba_update,-1.,1.)
            #             print(q_proba.numpy())
            # state update
            cur_inventory = self.block.call(q_proba, training=True)

            z_neg_alloc_sku_map = self.block.get_neg_alloc_sku_map()
            z_over_alloc_sku_map = self.block.get_over_alloc_sku_map()

            # calculate expected reward

            rs_bun_ava_sku_map = tf.reshape(
                z_bun_ava_sku_map,
                [1, len(self.block.payload.styles) * len(self.block.payload.colors)],
            )

            rs_neg_alloc_sku_map = tf.reshape(
                z_neg_alloc_sku_map,
                [
                    len(self.block.payload.shops),
                    len(self.block.payload.styles) * len(self.block.payload.colors),
                ],
            )

            rs_over_alloc_sku_map = tf.reshape(
                z_over_alloc_sku_map,
                [
                    len(self.block.payload.shops),
                    len(self.block.payload.styles) * len(self.block.payload.colors),
                ],
            )

            mip_target = tf.reshape(
                self.block.mip_target,
                [
                    len(self.block.payload.shops),
                    len(self.block.payload.styles) * len(self.block.payload.colors),
                ],
            )

            real_target = tf.reshape(
                z_tar_alloc,
                [
                    len(self.block.payload.shops),
                    len(self.block.payload.styles) * len(self.block.payload.colors),
                ],
            )

            # reshaped current inventory
            rs_ci = tf.reshape(
                cur_inventory,
                [
                    len(self.block.payload.shops),
                    len(self.block.payload.styles) * len(self.block.payload.colors),
                ],
            )

            mip_wgt_real_target = self.mip_sol_wgt * mip_target + (1 - self.mip_sol_wgt) * real_target

            if i == 0:
                
                q_expected_target = tf.reshape(mip_wgt_real_target, [1,-1])

                q_reward = rewardFunction(
                    mip_wgt_real_target, rs_ci, rs_bun_ava_sku_map, rs_neg_alloc_sku_map, rs_over_alloc_sku_map  
                )
            elif i > 0:

                q_expected_target = self.alpha * q_expected_target + (1 - self.alpha) * tf.reshape(mip_wgt_real_target, [1,-1])

                q_reward = self.alpha * q_reward + (1 - self.alpha) * rewardFunction(
                    mip_wgt_real_target, rs_ci, rs_bun_ava_sku_map, rs_neg_alloc_sku_map, rs_over_alloc_sku_map
                )

        self.prev_loss = tf.reduce_mean(tf.math.abs(q_reward)).numpy()

        q_cur_inventory = q_reward + z_cur_inventory_backup

        # reset states
        self.block.set_ava_bun(z_ava_bun_backup)
        self.block.set_cur_inv(z_cur_inventory_backup)
        self.block.set_shop_bun_alloc(z_shop_bundle_alloc_backup)
        self.block.set_shop_bun_proba(z_proba_backup)

        # this is derived from z_ava_bun, setting needed
        z_neg_alloc_sku_map = self.block.get_neg_alloc_sku_map()
        z_over_alloc_sku_map = self.block.get_over_alloc_sku_map()

        with tf.GradientTape() as tape0:
            # train model and get next state q_value



            rl_update = self.model(
                [
                    tf.cast(z_bun_lkup, tf.float32),
                    tf.cast(z_ava_bun_backup, tf.float32),
                    tf.cast(z_cur_inventory_backup, tf.float32),
                    tf.cast(z_neg_alloc_sku_map, tf.float32),
                    tf.cast(z_over_alloc_sku_map, tf.float32),
                    tf.cast(z_shop_bundle_alloc_backup, tf.float32),
                    tf.cast(z_proba_backup, tf.float32),
                    tf.cast(z_tar_alloc, tf.float32),
                ],
                training=True,
            )

            #             proba_update=tf.clip_by_value(tf.squeeze(proba_update),-1.,1.)
            rl_update = tf.squeeze(rl_update)


            
            print("update", rl_update.numpy().round(1))

            #             proba_update=tf.clip_by_value(tf.squeeze(proba_update),-1.,1.)
            #             proba_update=tf.squeeze(proba_update)
            #             print(proba_update.numpy())

            q_proba = tf.squeeze(z_proba_backup) + self.gamma * rl_update

            
            cur_inventory = self.block.call(q_proba, training=True)

            z_neg_bun_alloc_sku_map = self.block.get_neg_alloc_sku_map()
            z_over_bun_alloc_sku_map = self.block.get_over_alloc_sku_map()

            loss = self.loss_function(q_cur_inventory, cur_inventory)

            constraint_wgt = tf.reduce_sum(tf.abs(z_neg_bun_alloc_sku_map)) + tf.reduce_sum(tf.abs(z_over_bun_alloc_sku_map))

            grads = tape0.gradient(loss, self.model.trainable_variables)
            
            print(
                "loss",
                loss.numpy(),
                'constraint',
                constraint_wgt.numpy().round(3),
                "reward",
                tf.reduce_mean(tf.math.abs(q_reward)).numpy(),
            )
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if tf.reduce_mean(tf.math.abs(q_reward)).numpy() < 5:
                plt.figure(figsize=(15, 10))
                matrows = 1
                pltrows = 1

                cnt = 0
                plt.subplot(matrows, 4, pltrows).set_axis_off()
                pltrows += 1
                if cnt == 0:
                    plt.title("Reward", fontsize=16)
                plt.imshow(q_reward.numpy().reshape(-1, 3))
                plt.subplot(matrows, 4, pltrows).set_axis_off()
                pltrows += 1
                if cnt == 0:
                    plt.title("Q-Function", fontsize=16)
                plt.imshow(cur_inventory.numpy().reshape(-1, 3))
                plt.subplot(matrows, 4, pltrows).set_axis_off()
                pltrows += 1
                if cnt == 0:
                    plt.title("Target", fontsize=16)
                plt.imshow(self.block.mip_target.numpy().reshape(-1, 3))
                plt.subplot(matrows, 4, pltrows).set_axis_off()
                pltrows += 1
                if cnt == 0:
                    plt.title("Updated Q-Function", fontsize=16)
                plt.imshow(q_cur_inventory.numpy().reshape(-1, 3))

                plt.tick_params(
                    axis="both",  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,
                    labelleft=False,
                    labelbottom=False,
                )  # labels along the bottom edge are off

                plt.subplots_adjust(wspace=0, hspace=0.1)
                folder = "video"
                #             plt.savefig(folder + "/file%02d.png" % self.vid_count)
                #             if np.random.uniform()>0.9:
                plt.show()
                plt.close()
                self.vid_count += 1

        return 1
