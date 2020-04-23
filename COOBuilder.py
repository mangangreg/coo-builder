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
            self.terms[term] = len(self.terms)
        return self.terms[term]

    def _generate_ind2word(self):
        self.ind2word = {v:k for k,v in self.terms.items()}

    def word_lookup(self, ind):
        '''
        Looks up word

        ind - the index of a term in the self.terms dictionary
        returns the term (as a string)
        '''
        if not self.ind2word:
            self._generate_ind2word()
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
        return self.file_count, len(self.terms)

    def to_coo(self):
        '''
        Returns a coo_matrix (part of the sparse matrix family in scikit)
        '''

        # File buffers mean the data is never all in memory when you are building the matrix
        coord_i = np.frombuffer(self.i, dtype=np.int32)
        coord_j = np.frombuffer(self.j, dtype=np.int32)
        vals = np.frombuffer(self.data, dtype=np.int32)

        return sp.coo_matrix( (vals, (coord_i,coord_j)), shape = self.get_shape())

    def drop_columns(self, remove_idx):
        '''
        Remove columns from the matrix by id

        remove_idx (list) - a list of column indices to be removed
        '''
        if not self.ind2word:
            self._generate_ind2word()

        ind = 0

        while ind < len(self.data) :
            if self.j[ind] in remove_idx:
                self.i.pop(ind)
                self.j.pop(ind)
                self.data.pop(ind)
            else:
                ind +=1

        terms_new = {}
        id_new = 0
        for i in range(len(self.terms)):
            if i not in remove_idx:
                terms_new[self.ind2word[i]] = id_new
                id_new +=1


        self.terms = terms_new
        self._generate_ind2word()

    def drop_columns_by_terms(self, term_list):
        '''
        Remove columns from the matrix by term

        term_list (list) - a list of terms to be removed

        '''
        remove_idx = [self.terms[term] for term in term_list]
        self.drop_columns(remove_idx)
