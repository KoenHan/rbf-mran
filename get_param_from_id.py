import sqlite3
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--db_file')
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-id', '--trial_id', required=True)
    args = parser.parse_args()

    project_folder = './study/'+args.study_name
    conn = sqlite3.connect(project_folder+'/model/param.db')
    cur = conn.cursor()

    cur.execute('SELECT * FROM trial_params WHERE trial_id == '+str(int(args.trial_id)+1))

    param = {}
    for line in cur.fetchall() :
        param[line[2]] = line[3]
    for key in ['Nw', 'Sw', 'past_sys_input_num', 'past_sys_output_num']:
        param[key] = int(param[key])
    param['init_h'] = 0 # プログラムの都合上追記しとく
    param['E3'] = -1 # プログラムの都合上追記しとく

    param_file = project_folder+'/model/trial_id_'+str(args.trial_id)+'.yaml'
    with open(param_file, 'w') as f:
        yaml.dump(param, f, default_flow_style=False)
    print('Save as param file: ', param_file)

    cur.close()
    conn.close()