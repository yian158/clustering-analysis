import scipy.io as scio
import xlwt


def read_file(file_name):
    input_file = scio.loadmat(file_name)

    if isinstance(input_file, dict):
        key = 'dataSet'
        if key not in input_file:
            dict_key_list = list(input_file.keys())
            key = dict_key_list[-1]

        dataset = input_file[key]
        return dataset
    else:
        raise TypeError(file_name + 'is not a DICT')


def write_file(label, filename='cluster_label'):
    file = xlwt.Workbook()
    sheet = file.add_sheet('label')
    for i, l in enumerate(label):
        sheet.write(i, 0, str(l))
    file.save(filename + '.xls')