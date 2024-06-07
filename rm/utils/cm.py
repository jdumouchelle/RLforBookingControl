from pathlib import Path


def get_path(cfg, dtype, data_path, suffix='.pkl'):
	""" Gets path for a given problem config. """
	
	p = Path(data_path) / "cm"
	p.mkdir(parents=True, exist_ok=True)

	# p = p / f"{dtype}_wf{cfg.weight_factor}_vf{cfg.volume_factor}_d{cfg.deterministic}_nfo{cfg.n_final_obs}_nst_{cfg.n_train_samples}_nsv_{cfg.n_valid_samples}_sd{cfg.seed}{suffix}"
	p = p / f"{dtype}_wf{cfg.weight_factor}_vf{cfg.volume_factor}_d{cfg.deterministic}_nst_{cfg.n_train_samples}_nsv_{cfg.n_valid_samples}_sd{cfg.seed}{suffix}"

	return str(p)