import sqlite3

dbname = './model/param/mimo_optuna.db'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

# terminalで実行したSQL文と同じようにexecute()に書く
cur.execute('SELECT * FROM trial_params WHERE trial_id == 8')

# 中身を全て取得するfetchall()を使って、printする。
for line in cur.fetchall() :
    print(line[2], ': ', line[3])

cur.close()
conn.close()