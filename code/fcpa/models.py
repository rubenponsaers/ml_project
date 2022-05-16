def load_config(model):
	with open('config.yaml') as file:
		config = yaml.full_load(file)
	return config[model]