# Unmanned_Submersive_Vehicle
Gêmeo digital de alta fidelidade para Veículos de Submersíveis Autônomos. Implementa física de Fossen (6-DOF), filtros de Kalman Estendidos (EKF) e Aprendizado por Reforço Hierárquico (PPO) para navegação robusta em ambientes navais complexos.
# USV Digital Twin — Autonomia Naval via Aprendizado por Reforço Hierárquico

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositório contém o desenvolvimento de um **Gêmeo Digital (Digital Twin)** de alta fidelidade para um Veículo de Superfície Autônomo (USV). O projeto integra modelos hidrodinâmicos avançados, fusão de sensores estatística e Inteligência Artificial para resolver o problema de navegação autônoma em ambientes marítimos ruidosos.

## Destaques Técnicos

O diferencial deste projeto é a fidelidade da simulação, permitindo a transferência de políticas treinadas no virtual para o mundo real (**Sim-to-Real**) com mínima perda de performance.

* **Motor de Física (Fossen 6-DOF):** Implementação das equações de movimento baseadas em *Thor I. Fossen (2011)*, considerando massa adicionada, forças de Coriolis, arrasto quadrático e empuxo de lastro.
* **Percepção via EKF:** Filtro de Kalman Estendido para fusão de dados ruidosos (IMU, Barômetro e Sonar), garantindo uma estimativa de estado robusta mesmo com falhas sensoriais.
* **IA Hierárquica (HRL):** Arquitetura baseada em **PPO (Proximal Policy Optimization)** dividida em três níveis:
    1.  **N1 (Estabilização):** Controle de atitude e profundidade.
    2.  **N2 (Evasão):** Desvio de obstáculos baseado em sonar.
    3.  **N3 (Navegação):** Alcance de waypoints e planejamento de missão.
* **Algoritmo de Ganho Customizado:** Lógica de controle reformulada para estabilidade em dinâmicas não-lineares.

## Estrutura do Projeto

* `physics_engine.py`: O "coração" do simulador. Resolve as integrações via RK4.
* `control_engine.py`: Implementa controladores LQR e a lógica de ganhos customizada.
* `rl_controller.py`: Implementação própria do agente PPO e arquitetura hierárquica.
* `sensor_engine.py`: Simulação de hardware real (Open Echo Sonar, MS5837).
* `geometry_engine.py`: Cálculos de geometria de casco (Von Kármán ogive) e coeficientes.
* `visualization_server.py`: Ponte Flask + SocketIO para renderização em Three.js.

## 🛠️ Instalação e Uso

1. Clone o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/usv-digital-twin.git](https://github.com/seu-usuario/usv-digital-twin.git)
Instale as dependências:

Bash
pip install -r requirements.txt
Execute os testes de validação:

Bash
python physics_engine.py
python control_engine.py
Inicie o servidor de visualização:

Bash
python visualization_server.py
Metodologia Científica
O projeto foi construído sobre uma esteira de pesquisa iniciada no ICS (Inertial Control Sandbox), focando na transição de sistemas inerciais simples para dinâmicas navais complexas. O uso de Domain Randomization durante o treino da IA garante que o controlador seja resiliente a variações de densidade da água e ruído eletromagnético nos sensores.

Desenvolvido por Eduardo Souza Costa e Marcelo Henrique Valdierp.
