import tensorflow as tf
import keras.backend as K
import numpy as np
    
def contrastive_Loss_func(num_dist,temp=0.1):
    
    def loss(y_true, y_pred):
#        num_dist = tf.shape(y_pred)[1]
        view1 = y_pred[:, :num_dist+1]
        view2 = y_pred[:, num_dist+1:]
        
        loss_view1,loss_view2 = 0,0
        for i in range(num_dist+1):
            loss_view1 = loss_view1 + cmc_loss(view1[:,i], view2, i, num_dist)
            loss_view2 = loss_view2 + cmc_loss(view2[:,i], view1, i, num_dist)
        
        total_cost = (loss_view1 + loss_view2)/(num_dist+1)
        
        return(total_cost)
    return(loss)

   
def cmc_loss(anchor, view, channel,  num_dist, temp =0.1):
    
    anchor_norm = tf.math.l2_normalize(anchor, axis=-1)
    pos_view = view[:, channel]
    pos_view_norm = tf.math.l2_normalize(pos_view, axis=-1)
    
    pos_view_sim = tf.exp(tf.divide( tf.matmul(
        anchor_norm, tf.transpose(pos_view_norm)
    ), temp))
    
    pos_view_sim = tf.linalg.trace(pos_view_sim)
    
    neg_view_sum = 0
    for i in range(num_dist+1):
        if i != channel:
            neg_view = view[:, i]
            neg_view_norm = tf.math.l2_normalize(neg_view, axis=-1)
            neg_view_sim = tf.exp(tf.divide( tf.matmul(
                anchor_norm, tf.transpose(neg_view_norm)
            ), temp))
                
            neg_view_sim = tf.linalg.trace(neg_view_sim)
            neg_view_sum = neg_view_sum + neg_view_sim
    
    cost = -tf.math.log(tf.divide(pos_view_sim, (pos_view_sim + neg_view_sum)))
    
    return(tf.reduce_mean(cost))