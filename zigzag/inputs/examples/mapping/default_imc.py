mapping = {
    "default": {
        "core_allocation": 1,
        # "spatial_mapping": {"D1": ("OX", 25), "D2": (("FX", 3), ("FY", 3))},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        # "spatial_mapping_hint": {"D1": ["K", "OX"], "D2": ["C", "FX", "FY"]},
        "spatial_mapping_hint": {"D1": ["K"], "D3": ["K", "OX", "OY"]},
        # "spatial_mapping": {'D1': ('K', 4), 'D2': (('C', 3), ('FX', 3), ('FY', 3)), 'D3': (('OX', 2), ('OY', 2))},
        # "spatial_mapping": {'D1': ('K', 32), 'D2': ('C', 3), 'D3': ('FX', 3)},
    }
}
