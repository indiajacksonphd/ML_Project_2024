import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.image import ssim  # Make sure you import ssim if you are using it
from tensorflow.keras.metrics import Metric, MeanSquaredError

class MixedGlorotNormal(Initializer):
    def __init__(self, lamb=0.01):
        self.lamb = lamb
    
    def __call__(self, shape, dtype=None):
        fan_in, fan_out = tf.keras.initializers._compute_fans(shape)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return tf.random.normal(shape, mean=0.0, stddev=std, dtype=dtype) + tf.random.uniform(shape, minval=0, maxval=1/self.lamb, dtype=dtype)

    def get_config(self):
        return {'lamb': self.lamb}


class MixedHeNormal(Initializer):
    def __init__(self, lamb=0.01):
        self.lamb = lamb
    
    def __call__(self, shape, dtype=None):
        fan_in, _ = tf.keras.initializers._compute_fans(shape)
        std = np.sqrt(2.0 / fan_in)
        return tf.random.normal(shape, mean=0.0, stddev=std, dtype=dtype) + tf.random.uniform(shape, minval=0, maxval=1/self.lamb, dtype=dtype)

    def get_config(self):
        return {'lamb': self.lamb}
'''
class RMSE_SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='rmse_ssim', **kwargs):
        super(RMSE_SSIM, self).__init__(name=name, **kwargs)
        self.mse = MeanSquaredError()  # Now MeanSquaredError is recognized here
        self.rmse_ssim_value = self.add_weight(name='rmse_ssim', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        rmse = tf.sqrt(self.mse(y_true, y_pred))
        ssim_value = ssim(y_true, y_pred, max_val=1.0)  # Assuming y_true and y_pred are scaled between 0 and 1
        rmse_ssim = (rmse + (1.0 - ssim_value)) / 2.0
        self.rmse_ssim_value.assign(rmse_ssim)
        
    def result(self):
        return self.rmse_ssim_value
    
    def reset_states(self):
        self.rmse_ssim_value.assign(0.0)

class RMSE_SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='rmse_ssim', **kwargs):
        super(RMSE_SSIM, self).__init__(name=name, **kwargs)
        self.mse = MeanSquaredError()
        self.rmse_ssim_value = self.add_weight(name='rmse_ssim', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        rmse = tf.sqrt(self.mse(y_true, y_pred))
        ssim_value = ssim(y_true, y_pred, max_val=1.0)
        rmse_ssim = (rmse + (1.0 - ssim_value)) / 2.0
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            rmse_ssim = rmse_ssim * sample_weight
            self.rmse_ssim_value.assign_add(tf.reduce_sum(rmse_ssim) / tf.reduce_sum(sample_weight))
        else:
            self.rmse_ssim_value.assign_add(tf.reduce_mean(rmse_ssim))
    
    def result(self):
        return self.rmse_ssim_value
    
    def reset_states(self):
        self.rmse_ssim_value.assign(0.0)
'''

class RMSE_SSIM(Metric):
    def __init__(self, name='rmse_ssim', **kwargs):
        super(RMSE_SSIM, self).__init__(name=name, **kwargs)
        self.mse = tf.keras.metrics.MeanSquaredError()
        self.rmse_ssim_value = self.add_weight(name='rmse_ssim', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        mse = self.mse(y_true, y_pred)
        rmse = tf.sqrt(mse)
        ssim_value = ssim(y_true, y_pred, max_val=1.0)
        ssim_error = 1.0 - tf.abs(ssim_value)
        rmse_ssim = (rmse + ssim_error) / 2.0
        
        # Update metric state
        self.rmse_ssim_value.assign_add(tf.reduce_sum(rmse_ssim))
        
    def result(self):
        return self.rmse_ssim_value / self.num_samples
    
    def reset_states(self):
        self.rmse_ssim_value.assign(0.0)    