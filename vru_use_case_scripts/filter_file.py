def filter_file(input_file, output_file, pattern):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if pattern not in line:
                f_out.write(line)

# Example usage:
input_file = '/home/carma/Downloads/log-20240322-204234-Town04/Traffic_big.log'
output_file = '/home/carma/Downloads/log-20240322-204234-Town04/Traffic.log'
pattern_to_filter = 'AbstractSumoAmbassador:1219 - scheduled'

filter_file(input_file, output_file, pattern_to_filter)
