import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
from collections import defaultdict, Counter
import nltk
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
from tqdm import tqdm
import operator
from itertools import islice, count
import json
from io import StringIO

import builtins

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 199998
import builtins
bucket_name = '314047101_316294032'  # TODO check with talya

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, b_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self.client = storage.Client()
        self._bucket = self.client.bucket(b_name)
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            ' if the current file is full, close and open a new one.'
            if remaining == 0:
                self._f.close()
                file = self._f.name
                blob = self._bucket.blob(f"postings_gcp/{file}")
                blob.upload_from_filename(file)
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
        (1) `name`.pkl containing the global term stats (e.g. df).
    """
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}
        self.client = storage.Client()
        self._bucket = self.client.bucket(bucket_name)

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                blob = self._bucket.get_blob(f'postings_gcp/{f_name}')
                self._open_files[f_name] = blob.open('rb')

            f = self._open_files[f_name]
            f.seek(offset)
            n_read = builtins.min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this many bytes.
    TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

class InvertedIndex:
        def __init__(self, docs={}):
            """ Initializes the inverted index and add documents to it (if provided).
      Parameters:
      -----------
        docs: dict mapping doc_id to list of tokens
      """
            # stores document frequency per term
            self.df = Counter()
            # stores total frequency per term
            self.term_total = Counter()
            # stores posting list per term while building the index (internally),
            # otherwise too big to store in memory.
            self._posting_list = defaultdict(list)
            # mapping a term to posting file locations, which is a list of
            # (file_name, offset) pairs. Since posting lists are big we are going to
            # write them to disk and just save their location in this list. We are
            # using the MultiFileWriter helper class to write fixed-size files and store
            # for each term/posting list its list of locations. The offset represents
            # the number of bytes from the beginning of the file where the posting list
            # starts.
            self.posting_locs = defaultdict(list)

            self.doc_id_title = defaultdict(list)

            self.DL = defaultdict(list)

            for doc_id, tokens in docs.items():
                self.add_doc(doc_id, tokens)

        def add_doc(self, doc_id, tokens):
            """ Adds a document to the index with a given `doc_id` and tokens. It counts
          the tf of tokens, then update the index (in memory, no storage
          side-effects).
            """
            w2cnt = Counter(tokens)
            self.term_total.update(w2cnt)
            max_value = max(w2cnt.items(), key=operator.itemgetter(1))[1]
            # frequencies = {key: value/max_value for key, value in frequencies.items()}
            for w, cnt in w2cnt.items():
                self.df[w] = self.df.get(w, 0) + 1
                self._posting_list[w].append((doc_id, cnt))

        def write_index(self, base_dir, name):
            """ Write the in-memory index to disk. Results in the file:
            (1) `name`.pkl containing the global term stats (e.g. df).
            """
            self._write_globals(base_dir, name)

        def write(self, base_dir, name):
            """ Write the in-memory index to disk and populate the `posting_locs`
          variables with information about file location and offset of posting
          lists. Results in at least two files:
          (1) posting files `name`XXX.bin containing the posting lists.
          (2) `name`.pkl containing the global term stats (e.g. df).
      """
            #### POSTINGS ####
            self.posting_locs = defaultdict(list)
            with closing(MultiFileWriter(base_dir, name)) as writer:
                # iterate over posting lists in lexicographic order
                for w in sorted(self._posting_list.keys()):
                    self._write_a_posting_list(w, writer, sort=True)
            #### GLOBAL DICTIONARIES ####
            self._write_globals(base_dir, name)

        def _write_globals(self, base_dir, name):
            with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
                pickle.dump(self, f)

        def write_a_posting_list(b_w_pl):
            ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
      and writes it out to disk as files named {bucket_id}_XXX.bin under the
      current directory. Returns a posting locations dictionary that maps each
      word to the list of files and offsets that contain its posting list.
      Parameters:
      -----------
        b_w_pl: tuple
          Containing a bucket id and all (word, posting list) pairs in that bucket
          (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
      Return:
        posting_locs: dict
          Posting locations for each of the words written out in this bucket.
      '''
            posting_locs = defaultdict(list)
            bucket, list_w_pl = b_w_pl

            with closing(MultiFileWriter('.', bucket, bucket_name)) as writer:
                for w, pl in list_w_pl:
                    # convert to bytes
                    b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                                  for doc_id, tf in pl])
                    # write to file(s)
                    locs = writer.write(b)
                    # save file locations to index
                    posting_locs[w].extend(locs)
                file_name = writer._f.name
                blob = writer._bucket.blob(f"postings_gcp/{file_name}")
                blob.upload_from_filename(file_name)

            return posting_locs

        def __getstate__(self):
            """ Modify how the object is pickled by removing the internal posting lists
          from the object's state dictionary.
      """
            state = self.__dict__.copy()
            del state['_posting_list']
            return state

        def posting_lists_iter(self, query):
            """ A generator that reads one posting list from disk and yields
          a (word:str, [(doc_id:int, tf:int), ...]) tuple.
            """
            with closing(MultiFileReader()) as reader:
                for w in query:
                    posting_list = []
                    # read a certain number of bytes into variable b
                    if w in self.posting_locs:
                        locs = self.posting_locs[w]
                        b = reader.read(locs, self.df[w] * TUPLE_SIZE)

                        # convert the bytes read into `b` to a proper posting list.
                        for i in range(self.df[w]):
                            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                            posting_list.append((doc_id, tf))

                    yield w, posting_list

        @staticmethod
        def read_index(base_dir, name):
            with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
                return pickle.load(f)

        @staticmethod
        def delete_index(base_dir, name):
            path_globals = Path(base_dir) / f'{name}.pkl'
            path_globals.unlink()
            for p in Path(base_dir).rglob(f'{name}_*.bin'):
                p.unlink()

        def posting_lists_iter(self, query):
            """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.

        """
            with closing(MultiFileReader()) as reader:
                for w in query:
                    posting_list = []
                    # read a certain number of bytes into variable b
                    if w in self.posting_locs:
                        locs = self.posting_locs[w]
                        b = reader.read(locs, self.df[w] * TUPLE_SIZE)

                        # convert the bytes read into `b` to a proper posting list.
                        for i in range(self.df[w]):
                            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                            posting_list.append((doc_id, tf))

                    yield w, posting_list