import multiprocessing as mp

import time
def brute(transactions_path):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import random
    from model import predict # Функция, позволяет получить предсказание нейронки.

    import pytorch_lightning as pl
    import torch
    import pickle
    from model import process_for_nn, TransactionsDataset, get_dataloader, device, TransactionsRnn

    bins_path = "nn_bins.pickle" # путь до файла с бинами после тренировки модели (nn_bins.pickle)
    model_path = "nn_weights.ckpt" # путь до файла с весами нейронной сети (nn_weights.ckpt)


    train_target_path = "target_finetune.csv" # y - true target

    threshold = pd.read_csv(train_target_path).target.mean() # примерный threshold трейна​

    def create_dl(df, bins_path):
        df_transactions = df.dropna().assign(
                hour=lambda x: x.transaction_dttm.dt.hour,
                day=lambda x: x.transaction_dttm.dt.dayofweek,
                month=lambda x: x.transaction_dttm.dt.month,
                number_day=lambda x: x.transaction_dttm.dt.day,
            )

        with open(bins_path, "rb") as f:
            bins = pickle.load(f)
        features = bins.pop("features")

        df = process_for_nn(df_transactions, features, bins)
        dataset = TransactionsDataset(df)
        dataloader = get_dataloader(dataset, device, is_validation=True)
        return dataloader


    def predictor(dataloader, model):
        preds = []
        users = []
        for data, target in dataloader:
            y_pred = model(data)
            preds.append(y_pred.detach().cpu().numpy())
            users.append(target.detach().cpu().numpy())
        preds = np.concatenate(preds)
        users = np.concatenate(users)
        return pd.DataFrame({"user_id": users, "target": preds[:, 1]})


    def check_target_attack(user, target, attacked_target):
        if target.target_true.loc[target.user_id == user].values[0]:
            return target.target.loc[target.user_id == user].values[0] - attacked_target.target.loc[attacked_target.user_id == user].values[0]
        else:
            return attacked_target.target.loc[attacked_target.user_id == user].values[0] - target.target.loc[target.user_id == user].values[0]


    pl.seed_everything(26041999)
    model = TransactionsRnn()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with open(bins_path, "rb") as f:
        bins = pickle.load(f)

    random.seed(20230206)

    df_transactions = pd.read_csv(transactions_path, index_col=0)

    df_transactions['transaction_dttm'] = pd.to_datetime(df_transactions.transaction_dttm, format='%Y-%m-%d %H:%M:%S')


    df_transactions_attacked = df_transactions.copy()

    bins_path = "nn_bins.pickle"
    model_path = "nn_weights.ckpt"

    target = predict(transactions_path, bins_path, model_path)
    target['target_true'] = target.target.apply(lambda x: 0 if x <= threshold else 1)
    hero_users_top = target.loc[(target.target_true == 1)].sort_values('target')[-20:].user_id.values
    poor_users_top = target.loc[(target.target_true == 0)].sort_values('target')[:20].user_id.values

    one_idx = target.index[target.target > threshold]  # Эти пользователи похожи на Героя
    zero_idx = target.index[target.target <= threshold] # А эти на Неудачника

    users = target.user_id.values

    one_users = users[one_idx] # defolt - 1
    zero_users = users[zero_idx] # norm users - 0

    for user in tqdm(users, desc='Total'):
        df_transactions_user = df_transactions.loc[df_transactions.user_id == user].copy()
        user_score = 0
        best_copy_from = 829187
        best_i = 1
        if user in one_users:
            # для каждого из poor_users_top считаем лучишй скор в check_target_attack
            for copy_from in tqdm(poor_users_top, leave=False, desc='poor_users_top'):
                # сплитимся по рельсой по первым и последним транзакциям
                for i in range(1, 20):
                    idx_to_first = df_transactions_user.index[df_transactions_user.user_id == user][:5] # айдишники первых 5 транзакций юзера
                    idx_from_first = df_transactions.index[df_transactions.user_id == copy_from][:5*i:i] # айдишники первых 5 транзакций воннаби

                    idx_to_last = df_transactions_user.index[df_transactions_user.user_id == user][-5:] # айдишники последних 5 транзакций юзера
                    idx_from_last = df_transactions.index[df_transactions.user_id == copy_from][-5*i::i] # айдишники последних 5 транзакций воннаби

                    idx_to = idx_to_first.append(idx_to_last)
                    idx_from = idx_from_first.append(idx_from_last)

                    sign_to = np.sign(df_transactions.loc[idx_to, "transaction_amt"].values)
                    sign_from = np.sign(df_transactions.loc[idx_from, "transaction_amt"].values)
                    sign_mask = (sign_to == sign_from)

                    df_transactions_user.loc[idx_to[sign_mask], "mcc_code"] = df_transactions.loc[idx_from[sign_mask], "mcc_code"].values
                    df_transactions_user.loc[idx_to[sign_mask], "transaction_amt"] = df_transactions.loc[idx_from[sign_mask], "transaction_amt"].values

                    dataloader = create_dl(df_transactions_user, bins_path)

                    attacked_target = predictor(dataloader, model)

                    current_user_score = check_target_attack(user, target, attacked_target)
                    # если скор на пользователе улучшился, запоминаем пользователя и лучший шаг для сплита
                    if current_user_score >= user_score:
                        user_score = current_user_score
                        best_copy_from = copy_from
                        best_i = i

        else:
            # для каждого из hero_users_top считаем лучишй скор в check_target_attack
            for copy_from in tqdm(hero_users_top, leave=False, desc='hero_users_top'):
                # сплитимся по рельсой по первым и последним транзакциям
                for i in range(1, 20):
                    idx_to_first = df_transactions_user.index[df_transactions_user.user_id == user][:5] # айдишники первых 5 транзакций юзера
                    idx_from_first = df_transactions.index[df_transactions.user_id == copy_from][:5*i:i] # айдишники первых 5 транзакций воннаби

                    idx_to_last = df_transactions_user.index[df_transactions_user.user_id == user][-5:] # айдишники последних 5 транзакций юзера
                    idx_from_last = df_transactions.index[df_transactions.user_id == copy_from][-5*i::i] # айдишники последних 5 транзакций воннаби

                    idx_to = idx_to_first.append(idx_to_last)
                    idx_from = idx_from_first.append(idx_from_last)

                    sign_to = np.sign(df_transactions.loc[idx_to, "transaction_amt"].values)
                    sign_from = np.sign(df_transactions.loc[idx_from, "transaction_amt"].values)
                    sign_mask = (sign_to == sign_from)

                    df_transactions_user.loc[idx_to[sign_mask], "mcc_code"] = df_transactions.loc[idx_from[sign_mask], "mcc_code"].values
                    df_transactions_user.loc[idx_to[sign_mask], "transaction_amt"] = df_transactions.loc[idx_from[sign_mask], "transaction_amt"].values

                    dataloader = create_dl(df_transactions_user, bins_path)

                    attacked_target = predictor(dataloader, model)

                    current_user_score = check_target_attack(user, target, attacked_target)
                    # если скор на пользователе улучшился, запоминаем пользователя и лучший шаг для сплита
                    if current_user_score >= user_score:
                        user_score = current_user_score
                        best_copy_from = copy_from
                        best_i = i

        # проводим лучшую итоговую замену для текущего пользователя
        idx_to_first = df_transactions.index[df_transactions.user_id == user][:5]
        idx_from_first = df_transactions.index[df_transactions.user_id == best_copy_from][:5*best_i:best_i]

        idx_to_last = df_transactions.index[df_transactions.user_id == user][-5:]
        idx_from_last = df_transactions.index[df_transactions.user_id == best_copy_from][-5*best_i::best_i]

        idx_to = idx_to_first.append(idx_to_last)
        idx_from = idx_from_first.append(idx_from_last)

        sign_to = np.sign(df_transactions.loc[idx_to, "transaction_amt"].values)
        sign_from = np.sign(df_transactions.loc[idx_from, "transaction_amt"].values)
        sign_mask = (sign_to == sign_from)

        df_transactions_attacked.loc[idx_to[sign_mask], "mcc_code"] = df_transactions.loc[idx_from[sign_mask], "mcc_code"].values
        df_transactions_attacked.loc[idx_to[sign_mask], "transaction_amt"] = df_transactions.loc[idx_from[sign_mask], "transaction_amt"].values


    attacked_path = transactions_path[:-4] + '_attacked.csv'
    df_transactions_attacked.to_csv(attacked_path, index=False)

if __name__ == '__main__':
    data = [(f'datasets/sample_submission{i}.csv') for i in range(4)]
    # # n_workers = mp.cpu_count() * 2
    n_workers = 4
    print(n_workers)

    timer = time.time()

    pool = mp.Pool(processes=n_workers)
    pool.map(brute, data)
    print(f'time = {time.time() - timer}')
