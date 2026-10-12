import os

privacy_file = "apps/backend/src/privacy/fl_simulator.py"
try:
    with open(privacy_file, 'r') as f:
        content = f.read()
    
    # Simple search and replace for numpy -> tenseal if available
    if "import tenseal" not in content:
        imports = """
import numpy as np
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
"""
        content = content.replace("import numpy as np", imports)
        
        he_context_str = """
from dataclasses import dataclass, field

@dataclass
class HEContext:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: list = field(default_factory=lambda: [60, 40, 40, 60])
    scale: float = 2**40

class HomomorphicAggregator:
    def setup_context(self, he_ctx: HEContext):
        if not TENSEAL_AVAILABLE: return None
        return ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=he_ctx.poly_modulus_degree, coeff_mod_bit_sizes=he_ctx.coeff_mod_bit_sizes)
"""
        content = content + "\n\n" + he_context_str
        
        with open(privacy_file, 'w') as f:
            f.write(content)
        print("Updated FL Simulator with TenSEAL HEContext")
except Exception as e:
    print(f"Failed to update fl_simulator: {e}")
