# Guia de Integração Docker — USV Digital Twin

## 🚀 Quick Start

### Build da imagem
```bash
docker build -t usv-digital-twin .
```

### Executar serviço de visualização
```bash
docker run -p 5000:5000 \
  -v $(pwd)/training_runs:/app/training_runs \
  -v $(pwd)/checkpoints:/app/checkpoints \
  usv-digital-twin server
```

### Executar benchmark
```bash
docker run \
  -v $(pwd)/training_runs:/app/training_runs \
  -v $(pwd)/checkpoints:/app/checkpoints \
  usv-digital-twin benchmark --scenario mission --steps 2000
```

## 🐳 Docker Compose (Recomendado)

### Iniciar todos os serviços
```bash
docker-compose up -d
```

### Parar serviços
```bash
docker-compose down
```

### Ver logs
```bash
docker-compose logs -f server
docker-compose logs -f benchmark
```

## 📦 Serviços Disponíveis

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `server` | 5000 | Interface web Flask + replay visualization |
| `benchmark` | — | CLI para testes de performance |
| `training` | — | Pipeline de treinamento RL |

## 🔧 Modos de Execução

### 1. Servidor (Padrão)
```bash
docker run -p 5000:5000 usv-digital-twin server
```
Acesso: http://localhost:5000

### 2. Benchmark
```bash
docker run usv-digital-twin benchmark --scenario mission
docker run usv-digital-twin benchmark --scenario stability
```

### 3. Comando customizado
```bash
docker run usv-digital-twin python mpc_controller.py --config custom.json
```

## 📁 Volumes Persistentes

A imagem monta dois diretórios:
- `/app/training_runs` — Resultados de simulações e replays
- `/app/checkpoints` — Modelos RL salvos

Use `-v` para mapear para seu sistema local:
```bash
docker run -v /seu/caminho/training_runs:/app/training_runs usv-digital-twin
```

## 🌐 Networking (Multi-container)

Se adicionar mais serviços (DB, cache, etc):
```bash
docker-compose up -d
# Os serviços se comunicam pelo nome do serviço
# Ex: service1 → http://service2:port
```

## 📊 Monitoramento

### Ver consumo de recursos
```bash
docker stats usv-server
```

### Inspecionar variáveis de ambiente
```bash
docker inspect usv-server | grep ENV
```

## 🛠️ Melhorias Sugeridas

1. **Multi-stage build** (para imagem menor)
2. **Health checks** no docker-compose
3. **Arquivo .env** para configurações
4. **.env.example** para documentação
5. **Nginx reverseproxy** para produção

Deseja implementar alguma dessas melhorias?
