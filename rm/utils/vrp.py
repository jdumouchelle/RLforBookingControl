from pathlib import Path


def get_path(cfg, dtype, data_path, suffix='.pkl'):
	""" Gets path for a given problem config. """
	
	p = Path(data_path) / "vrp"
	p.mkdir(parents=True, exist_ok=True)

	#p = p / f"{dtype}_l{cfg.n_locations}_p{cfg.n_products}_t{cfg.n_periods}_v{cfg.n_vehicles}_sd{cfg.seed}{suffix}"
	p = p / f"{dtype}_l{cfg.n_locations}_p{cfg.n_products}_t{cfg.n_periods}_v{cfg.n_vehicles}_lf_{cfg.load_factor}_nst_{cfg.n_train_samples}_nsv_{cfg.n_valid_samples}_sd{cfg.seed}{suffix}"
			
	return str(p)