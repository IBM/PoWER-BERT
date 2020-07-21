import argparse


def retention_config_parser(rentention_config, num_layers, seq_len):

    """
    Parse the retention configuration provided by the user.
    Also, check the provided rentention config for correctness.
    """

    retention_config_list = []
    for length in rentention_config.split(","):
        try:
            int_length = int(length)
            assert int_length <= 512
            assert int_length <= seq_len
            assert len(retention_config_list) == 0 or int_length <= retention_config_list[-1]
        except:
            raise argparse.ArgumentTypeError(
                "Invalid sequence length %s. The sequence lengths have to be <= 512 and sequence length individually, and sorted in non-increasing order."
                % (length))
        retention_config_list.append(int_length)
    try:
        assert len(retention_config_list) == num_layers
    except AssertionError:
        raise argparse.ArgumentTypeError(
            "Length of the retention config list should be equal to the number of layers in the model.")
    return retention_config_list

