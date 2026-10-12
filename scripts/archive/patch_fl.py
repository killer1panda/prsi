with open('apps/backend/src/privacy/fl_simulator.py', 'r') as f:
    content = f.read()
content = content.replace("self.{k: v for k, v in model.state_dict().items() if 'lora' in k}", "{k: v for k, v in self.model.state_dict().items() if 'lora' in k}")
with open('apps/backend/src/privacy/fl_simulator.py', 'w') as f:
    f.write(content)
