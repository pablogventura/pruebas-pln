"""Generate sentences from an n-gram model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle
from languagemodeling.ngram import NGramGenerator


if __name__ == '__main__':
    opts = docopt(__doc__)

    # parse args
    n = int(opts['-n'])

    # load the trained model
    filename = opts['-i']
    trained_model = pickle.load(open(filename,'rb'))

    tr_model = NGramGenerator(trained_model)

    print("\r\nGenerando sentencias\r\n") 
    for i in range(n):
       print(' '.join(tr_model.generate_sent()),"\r\n")
    
