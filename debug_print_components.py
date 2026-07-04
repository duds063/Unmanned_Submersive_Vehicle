from train_barrel_roll_rl import BarrelRollConfig, BarrelRollTrainer

cfg = BarrelRollConfig(episodes=2, episode_steps=256, seed=99)
trainer = BarrelRollTrainer(cfg)
summary = trainer.run_episode(0, training=True)
print('SUMMARY:', summary)
print('REWARD_COMPONENTS:', summary.reward_components)
