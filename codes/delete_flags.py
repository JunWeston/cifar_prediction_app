import tensorflow as tf

def delete_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        
delete_all_flags(tf.flags.FLAGS)


