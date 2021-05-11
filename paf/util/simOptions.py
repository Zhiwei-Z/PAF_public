from optparse import OptionParser
import json
from cs285.scripts.default import DEFAULT_CONFIG


def get_options(_args=None):
    parser = OptionParser()

    parser.add_option("--config", "-c",
                      action="store", metavar="STRING", dest="config_file", default=None,
                      help="""The json config file that many of the config settings can be parsed from""")

    parser.add_option("--tuning", "-t",
                      action="store", metavar="STRING", dest="tuning", default=None,
                      help="""The json config file that many of the config settings can be parsed from json for 
                      tuning parameters""")

    parser.add_option("--random_seeds", "-r",
                      action="store", dest="random_seeds", default=1,
                      type=int,
                      metavar="INTEGER",
                      help="Number of seeds")

    parser.add_option("--num_parallel", "-p",
                      action="store", dest="num_parallel", default=1,
                      type=int,
                      metavar="INTEGER",
                      help="Number of parallel instances")

    parser.add_option("--exp_name", "-n",
                      action="store", metavar="STRING", dest="exp_name", default=None,
                      help="The experiment name")


    if _args is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(_args)

    ### convert to vars
    options = vars(options)

    ### Convert to dictionary
    options = process_options(options)
    return options



def process_options(options):
    ### Convert options into a dictionary
    file = open(options['config_file'])
    settings = json.load(file)
    settings = deep_update_dict(settings, DEFAULT_CONFIG)
    file.close()
    
    for option in options:
        if options[option] is not None:
            print("Updating option: ", option, " = ", options[option])
            settings[option] = options[option]
            try:
                settings[option] = json.loads(settings[option])
            except Exception as e:
                pass  # dataTar.close()
            if options[option] == 'true':
                settings[option] = True
            elif options[option] == 'false':
                settings[option] = False
    
    return settings


def deep_update_dict(fr, to):
    """ update dict of dicts with new values """
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

