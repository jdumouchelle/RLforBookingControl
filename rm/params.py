from types import SimpleNamespace


#-----------------------------------------#
#        Airline Cargo Management         #
#-----------------------------------------#

cm_sl_2_2 = SimpleNamespace(
    single_leg=True,
    weight_factor=0.5,
    volume_factor=0.5,
    deterministic=False,
    n_final_obs=50,
    n_samples=10000,        # for data generation
    n_train_samples=5000,   # for data generation
    n_valid_samples=1000,   # for data generation
    time_limit=60,          # for data generation
    mip_gap=0.0001,         # for data generation
    seed=1234,
)

cm_sl_2_1 = SimpleNamespace(
    single_leg=True,
    weight_factor=0.5,
    volume_factor=1.0,
    deterministic=False,
    n_final_obs=50,
    n_samples=10000,        # for data generation
    n_train_samples=5000,   # for data generation
    n_valid_samples=1000,   # for data generation
    time_limit=60,          # for data generation
    mip_gap=0.0001,         # for data generation
    seed=1234,
)

cm_sl_1_2 = SimpleNamespace(
    single_leg=True,
    weight_factor=1.0,
    volume_factor=0.5,
    deterministic=False,
    n_final_obs=50,
    n_samples=10000,        # for data generation
    n_train_samples=5000,   # for data generation
    n_valid_samples=1000,   # for data generation
    time_limit=60,          # for data generation
    mip_gap=0.0001,         # for data generation
    seed=1234,
)

cm_sl_1_1 = SimpleNamespace(
    single_leg=True,
    weight_factor=1.0,
    volume_factor=1.0,
    deterministic=False,
    n_final_obs=50,
    n_samples=10000,        # for data generation
    n_train_samples=5000,   # for data generation
    n_valid_samples=1000,   # for data generation
    time_limit=60,          # for data generation
    mip_gap=0.0001,         # for data generation
    seed=1234,
)


#-----------------------------------------#
#         Vehicle Routing Problem         #
#-----------------------------------------#

vrp_l4_t20_v2_llf = SimpleNamespace(
    # problem
    n_locations = 4,
    n_products = 1,
    n_periods = 20,
    n_vehicles = 2,
    load_factor = 1.1,
    location_bounds = [0,10],
    added_vehicle_cost=100,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 4,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l10_t30_v4_llf = SimpleNamespace(
    # problem
    n_locations = 10,
    n_products = 1,
    n_periods = 30,
    n_vehicles = 3,
    load_factor = 1.1,
    location_bounds = [0,10],
    added_vehicle_cost=100,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 10,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l15_t50_v4_llf = SimpleNamespace(
    # problem
    n_locations = 15,
    n_products = 1,
    n_periods = 50,
    n_vehicles = 3,
    load_factor = 1.1,
    location_bounds = [0,10],
    added_vehicle_cost=250,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 15,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l50_t100_v4_llf = SimpleNamespace(
    # problem
    n_locations = 50,
    n_products = 1,
    n_periods = 100,
    n_vehicles = 3,
    load_factor = 1.1,
    location_bounds = [0,50],
    added_vehicle_cost=600,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 50,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l4_t20_v2_hlf = SimpleNamespace(
    # problem
    n_locations = 4,
    n_products = 1,
    n_periods = 20,
    n_vehicles = 2,
    load_factor = 1.8,
    location_bounds = [0,10],
    added_vehicle_cost=100,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 4,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l10_t30_v4_hlf = SimpleNamespace(
    # problem
    n_locations = 10,
    n_products = 1,
    n_periods = 30,
    n_vehicles = 3,
    load_factor = 1.8,
    location_bounds = [0,10],
    added_vehicle_cost=100,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 10,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l15_t50_v4_hlf = SimpleNamespace(
    # problem
    n_locations = 15,
    n_products = 1,
    n_periods = 50,
    n_vehicles = 3,
    load_factor = 1.8,
    location_bounds = [0,10],
    added_vehicle_cost=250,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 15,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)

vrp_l50_t100_v4_hlf = SimpleNamespace(
    # problem
    n_locations = 50,
    n_products = 1,
    n_periods = 100,
    n_vehicles = 3,
    load_factor = 1.8,
    location_bounds = [0,50],
    added_vehicle_cost=600,

    # filo
    parser = 'E',       
    tolerance = 0.01,
    granular_neighbors = 1,
    cache = 50,
    routemin_iterations = 1000,
    coreopt_iterations = 100000,
    granular_gamma_base = 0.250,
    granular_delta = 0.500,
    shaking_lower_bound = 0.375,
    shaking_upper_bound = 0.850,

    # data geneartion
    n_train_samples = 5000,
    n_valid_samples = 1000,

    seed=1234,
)
