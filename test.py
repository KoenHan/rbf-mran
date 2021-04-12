import optuna

def obj(trial):
    x = trial.suggest_uniform('x', -10, 10)
    score = (x - 2)**2
    print('x: %1.3f, score: %1.3f' % (x, score))
    return score

study = optuna.create_study()
study.optimize(obj, n_trials=100)

print(study.best_params)
print(study.best_value)
print(study.best_trial)

