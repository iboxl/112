mapping = {
    "default": {
        "core_allocation": 0,
        # "spatial_mapping": {"D1": ("OX", 25), "D2": (("FX", 3), ("FY", 3))},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
        # "spatial_mapping_hint": {"D1": ["K", "OX"], "D2": ["C", "FX", "FY"]},
        "spatial_mapping_hint": {"D1": ["K"], "D3": ["K", "OX", "OY"]},
        # "spatial_mapping": {'D1': ('K', 4), 'D2': ('FX', 3), 'D3': [('K', 2),('OX',2)]},
    }
}
