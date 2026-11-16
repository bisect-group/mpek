def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU, the real ProtT5 weights and the EMULaToR data; run by verify_equivalence.sh --gpu")
