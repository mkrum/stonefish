#! /usr/bin/env python

import fileinput

if __name__ == '__main__':
    lines = [line for line in fileinput.input()]
    rouge = list(filter(lambda x: '>>' in x, lines))[0]

    rouge = rouge.split("/")
    rouge_two = float(rouge[3])
    rouge_l = float(rouge[4])

    bleu = list(filter(lambda x: 'Bleu_' in x, lines))
    bleu = [l.strip() for l in bleu]

    bleu_three = float(bleu[2].split(': ')[1])
    bleu_four = float(bleu[3].split(': ')[1])

    meteor = list(filter(lambda x: 'METEOR:' in x, lines))[0]
    meteor = float(meteor.split(': ')[1])

    cider = list(filter(lambda x: 'CIDEr:' in x, lines))[0]
    cider = float(cider.split(': ')[1])

    print(f'{rouge_two},{rouge_l},{bleu_three},{bleu_four},{meteor},{cider}')
