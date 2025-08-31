import vpi

available_backends = vpi.get_available_backends()

print("Available VPI Backends:")
if vpi.Backend.CUDA in available_backends:
    print("- CUDA")
if vpi.Backend.OFA in available_backends:
    print("- OFA")
if vpi.Backend.PVA in available_backends:
    print("- PVA")
if vpi.Backend.VIC in available_backends:
    print("- VIC")
