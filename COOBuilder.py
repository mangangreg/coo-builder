class COOBuilder():
    '''
    An object used to build a COO matrix.
    Based on https://maciejkula.github.io/2015/02/22/incremental-construction-of-sparse-matrices/
    But allowing for unknown shape
    '''

    def __init__(self):

        # i,j,data storage as array.array datatype (stores more efficiently than a standard python array), the 'I' just indicates it's an indicator, see table in help(array.array)
        self.i = array.array('I') # Assumes the file index is an int AND goes from 0 to k for k being the number of docs
        self.j = array.array('I')
        self.data = array.array('I')

        # The dictionary of common terms (which maps a term to an integer that coresponds to hypothetical column index) and the counter for # of common terms
        self.terms = {}
        self.term_count = 0

        #file count tracker
        self.file_count = 0

        # Used for word lookup, initialise as None so that its only built when needed
        self.ind2word = None

    def _get_term_id(self, term):
        '''
        An interface with the terms dictionary. Returns the term id, and if it doesn't exist will add it to the dictionary

        term (string) - the word/term being added
        '''
        if term not in self.terms:
            self.terms[term] = self.term_count
            self.term_count += 1
        return self.terms[term]

    def word_lookup(self, ind):
        '''
        Looks up word

        ind - the index of a term in the self.terms dictionary
        returns the term (as a string)
        '''
        if self.ind2word == None:
            self.ind2word = {v:k for k,v in builder.terms.items()}
        return self.ind2word[ind]

    def add_doc_counter(self, file_ind, term_counter):
        '''
        Adds the term count data for a new file

        file_ind (int) - the file index of the document
        term_counter (Counter)- a Counter object with the term counts for the file
        '''
        # Assumes this file has not been entered previously
        self.file_count += 1

        # Adds the co-ordinates and data
        for term, count in term_counter.items():
            self.i.append(file_ind)
            self.j.append(self._get_term_id(term))
            self.data.append(count)

    def get_shape(self):
        '''
        Get the shape of the hypothetical matrix
        '''
        return self.file_count, self.term_count

    def to_coo(self):
        '''
        Returns a coo_matrix (part of the sparse matrix family in scikit)
        '''

        # File buffers mean the data is never all in memory when you are building the matrix
        coord_i = np.frombuffer(self.i, dtype=np.int32)
        coord_j = np.frombuffer(self.j, dtype=np.int32)
        vals = np.frombuffer(self.data, dtype=np.int32)

        return sp.coo_matrix( (vals, (coord_i,coord_j)), shape = self.get_shape())
