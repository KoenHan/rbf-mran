import sqlite3
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--database_file', required=True)
    parser.add_argument('-id', '--trial_id', required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.database_file)
    cur = conn.cursor()

    cur.execute('SELECT * FROM trial_params WHERE trial_id == '+args.trial_id)

    for line in cur.fetchall() :
        print(line[2], ': ', line[3])

    cur.close()
    conn.close()