import json
import argparse


def main(args):
    with open(args.json_control) as f:
        dict_control = json.load(f)
    with open(args.json_exp) as f:
        dict_exp = json.load(f)

    result = []
    
    for k, v_exp in dict_exp.items():
        v_control = dict_control[k]
        diff = v_exp - v_control
        result.append(
            {
                'filename':k,
                'diff':diff,
                'exp':v_exp,
                'control':v_control
            }
        )
    result.sort(key=lambda x:-x['diff'])
    with open(args.save_path, 'w') as f:
        json.dump(result, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_control', default='temp/result0.json', help='control group')
    parser.add_argument('--json_exp', default='temp/result1.json', help='experimental group')
    parser.add_argument('--save_path', default='temp/compare.json', help='path to save result')
    args = parser.parse_args()
    main(args)