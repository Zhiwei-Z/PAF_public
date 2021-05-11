def get_env_policy(env_name):
    if env_name == "ant":
        return "cs285/policies/experts/Ant.pkl"
    if env_name == "hc":
        return "cs285/policies/experts/HalfCheetah.pkl"
    if env_name == "hopper":
        return "cs285/policies/experts/Hopper.pkl"
    if env_name == "humanoid":
        return "cs285/policies/experts/Humanoid.pkl"
    if env_name == "walker":
        return "cs285/policies/experts/Walker2d.pkl"


def get_env_data(env_name):
    if env_name == "ant":
        return "cs285/expert_data/expert_data_Ant-v2.pkl"
    if env_name == "hc":
        return "cs285/expert_data/expert_data_HalfCheetah-v2.pkl"
    if env_name == "hopper":
        return "cs285/expert_data/expert_data_Hopper-v2.pkl"
    if env_name == "humanoid":
        return "cs285/expert_data/expert_data_Humanoid-v2.pkl"
    if env_name == "walker":
        return "cs285/expert_data/expert_data_Walker2d-v2.pkl"


def get_formal_env_name(env_name):
    if env_name == "ant":
        return "Ant-v2"
    if env_name == "hc":
        return "HalfCheetah-v2"
    if env_name == "hopper":
        return "Hopper-v2"
    if env_name == "humanoid":
        return "Humanoid-v2"
    if env_name == "walker":
        return "Walker2d-v2"


def config_by_env(env_name):
    return {"env_name": get_formal_env_name(env_name),
            "expert_policy_file": get_env_policy(env_name),
            "expert_data": get_env_data(env_name)}


