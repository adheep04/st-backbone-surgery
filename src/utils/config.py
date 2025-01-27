def get_config():
    return {
        'max_steps' : 18000,
        'lr' : 0.2,
        'factor' : 0.75,
        'freq' : 10,
        'alpha' : 1,
        'beta' : 1e-6,
        'trial' : 'style_trial_x',
        'patience' : 6,
        'content_layer' : 1,
        'style_layer' : 6,
        'content_path' : './temp/data/content_img.jpg',
        'style_path' : './temp/data/style_img.jpg',
        'wl' : [0.2, 0.2, 0.2, 0.2, 0.2]
    }