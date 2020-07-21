import keras.backend as K


def metric_cor (y_true, y_pred):

    n = K.sum(K.ones_like(y_true))
    sum_x = K.sum(y_true)
    sum_y = K.sum(y_pred)
    sum_x_sq = K.sum(K.square(y_true))
    sum_y_sq = K.sum(K.square(y_pred))
    psum = K.sum(y_true * y_pred)
    num = psum - (sum_x * sum_y / n)
    den = K.sqrt((sum_x_sq - K.square(sum_x) / n) *  (sum_y_sq - K.square(sum_y) / n))
    return (num / den)
