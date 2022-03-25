AlphaSparse documentation
=================================================================

This document provides a detailed stage function interface description
of the AlphaSparse sparse matrix algorithm library, including mv, mm and
spmmd, etc. The function will perform the corresponding algorithm
processing according to the sparse matrix format. The specific format of
the sparse matrix will not be reflected in the interface name and the
reception parameters.

GENERAL
-------

Function return value
~~~~~~~~~~~~~~~~~~~~~~~~

The function has several return values, indicating whether the function
is executed successfully, as shown below:

+--------------------------------+-------------------------------------+
| return value                   | Return value meaning                |
+================================+=====================================+
|                                | Successful operation                |
| ``ALPHA_SPARSE_STATUS_SUCCESS``|                                     |
+--------------------------------+-------------------------------------+
| ``ALPHA_S                      | The matrix is not initialized       |
| PARSE_STATUS_NOT_INITIALIZED`` |                                     |
+--------------------------------+-------------------------------------+
| ``ALPH                         | Internal space allocation failed    |
| A_SPARSE_STATUS_ALLOC_FAILED`` |                                     |
+--------------------------------+-------------------------------------+
| ``ALPHA                        | The input parameter contains an     |
| _SPARSE_STATUS_INVALID_VALUE`` | illegal value                       |
+--------------------------------+-------------------------------------+
| ``ALPHA_SP                     | Execution failed                    |
| ARSE_STATUS_EXECUTION_FAILED`` |                                     |
+--------------------------------+-------------------------------------+
| ``ALPHA_                       | Algorithm implementation error      |
| SPARSE_STATUS_INTERNAL_ERROR`` |                                     |
+--------------------------------+-------------------------------------+
| ``ALPHA                        | The requested operation cannot be   |
| _SPARSE_STATUS_NOT_SUPPORTED`` | supported                           |
+--------------------------------+-------------------------------------+

.. list-table:: Frozen Delights!
   :widths: 15 10 30
   :header-rows: 1

   * - Treat
     - Quantity
     - Description
   * - Albatross
     - 2.99
     - On a stick!
   * - Crunchy Frog
     - 1.49
     - If we took the bones out, it wouldn't be
       crunchy, now would it?
   * - Gannet Ripple
     - 1.99
     - On a stick!

Basic operations of sparse matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithm library currently supports the direct creation of CSR, COO,
CSC and BSR sparse matrix format, and conversion from COO to CSR, CSC,
BSR, DIA and SKY, the interface supports four data formats.

alphasparse_create_coo
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_create_coo( 
       alphasparse_matrix_t *A, 
       const alphasparse_index_base_t indexing, 
       const ALPHA_INT rows, 
       const ALPHA_INT cols, 
       const ALPHA_INT nnz, 
       ALPHA_INT *row_indx, 
       ALPHA_INT *col_indx, 
       ALPHA_Number * values)

The ``alphasparse_?_create_coo creates`` a sparse matrix in ``COO``
matrix format. The matrix size is ``m*k`` and stored in variable ``A``.
“``?``” indicates the data format. It corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, and ``c`` corresponds to float complex, that is
single-precision complex number, ``z``\ corresponds to double complex,
that is double-precision complex number, other parameters are shown
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| A            | COO format                                            |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, index starts with                       |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, index starts with 1                       |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of matrix ``A``                        |
+--------------+-------------------------------------------------------+
| cos          | Number of columns of matrix ``A``                     |
+--------------+-------------------------------------------------------+
| nnz          | The number of non-zero elements of matrix ``A``       |
+--------------+-------------------------------------------------------+
| row_indx     | The row coordinate index of each non-zero element,    |
|              | the length is ``nnz``                                 |
+--------------+-------------------------------------------------------+
| col_indx     | The column coordinate index of each non-zero element, |
|              | the length is ``nnz``                                 |
+--------------+-------------------------------------------------------+
| values       | Store the values of non-zero elements in matrix ``A`` |
|              | in any order, with a length of ``nnz``                |
+--------------+-------------------------------------------------------+

alphasparse_create_csc
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_create_csc( 
       alphasparse_matrix_t *A,
       const alphasparse_index_base_t indexing, 
       const ALPHA_INT rows, 
       const ALPHA_INT cols, 
       ALPHA_INT *cols_start, 
       ALPHA_INT *cols_end, 
       ALPHA_INT *row_indx, 
       ALPHA_Number * values)

The ``alphasparse_?_create_csc`` is used to create a sparse matrix in
``CSC`` matrix format. The matrix size is ``m*k`` and is stored in
variable ``A``. “``?``” indicates the data format. It corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, and ``c`` corresponds to float complex, that is
single-precision complex number, ``z``\ corresponds to double complex,
that is double-precision complex number, other parameters are shown
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| A            | ``CSC`` format                                        |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input array,     |
|              | There are the following options:                      |
|              | \ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on 0        |
|              | addressing, index starts with 0                       |
|              | \ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1         |
|              | addressing, index starts with 1                       |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of matrix ``A``                        |
+--------------+-------------------------------------------------------+
| cols         | Number of columns of matrix ``A``                     |
+--------------+-------------------------------------------------------+
| cols_start   | The length is at least m, contains the index of each  |
|              | column of the matrix, ``cols_start[i] – ind`` is the  |
|              | starting index of the ``i-th`` column in values and   |
|              | ``row_indx``; When the input array is addressed based |
|              | on 0, the value of ind is 0;When addressed based on   |
|              | 1, the value of ``ind`` is 1.                         |
+--------------+-------------------------------------------------------+
| cols_end     | The length is at least m, contains the index of each  |
|              | column of the matrix, ``cols_end[i] – ind`` is the    |
|              | end position of the i-th column in values and         |
|              | ``row_indx``; When the input array is addressed based |
|              | on 0, the value of ``ind`` is 0;When addressed based  |
|              | on 1, the value of ``ind`` is 1.                      |
+--------------+-------------------------------------------------------+
| row_indx     | When addressing based on 1, the array contains the    |
|              | row index of each non-zero element of ``A +1``. When  |
|              | addressing based on 0, the array contains the row     |
|              | index of each non-zero element of ``A`` matrix; The   |
|              | length is at least ``cols_end[cols-1] – ind``.When    |
|              | the input array is addressed based on 0, the value of |
|              | ind is 0;When addressed based on 1, the value of ind  |
|              | is 1                                                  |
+--------------+-------------------------------------------------------+
| values       | Store the value of the non-zero element in the matrix |
|              | ``A``, length is equal to the length of ``row_indx``  |
+--------------+-------------------------------------------------------+

alphasparse_create_csr
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_create_csr(
       alphasparse_matrix_t *A, 
       const alphasparse_index_base_t indexing, 
       const ALPHA_INT rows, 
       const ALPHA_INT cols, 
       ALPHA_INT *rows_start, 
       ALPHA_INT *rows_end, 
       ALPHA_INT *col_indx, 
       ALPHA_Number * values)

The ``alphasparse_?_create_csr`` is used to create a sparse matrix in
``CSR`` matrix format. The matrix size is ``m*k`` and is stored in
variable ``A``. “``?``” indicates the data format. It corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, and ``c`` corresponds to float complex, that is
single-precision complex number, ``z``\ corresponds to double complex,
that is double-precision complex number, other parameters are shown
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| A            | CSR format                                            |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, the index starts with                   |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, the index starts with 1                   |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of matrix ``A``                        |
+--------------+-------------------------------------------------------+
| cols         | Number of columns of matrix ``A``                     |
+--------------+-------------------------------------------------------+
| rows_start   | The length is at least m, contains the index of each  |
|              | column of the matrix, ``rows_start[i] – ind`` is the  |
|              | starting index of the i-th column in values and       |
|              | ``col_indx``; when the input array is addressed based |
|              | on 0, the value of ind is 0;when addressed based on   |
|              | 1, the value of ind is 1.                             |
+--------------+-------------------------------------------------------+
| rows_end     | The length is at least m, contains the index of each  |
|              | column of the matrix, ``rows_end[i] – ind`` is the    |
|              | ``i-th`` column in values and the end position in     |
|              | ``col_indx``;when the input array is addressed based  |
|              | on 0, the value of ind is 0;when addressed based on   |
|              | 1, the value of ind is 1;                             |
+--------------+-------------------------------------------------------+
| col_indx     | When addressing based on 1, the array contains the    |
|              | row index of each non-zero element of ``A +1``. When  |
|              | addressing based on 0, the array contains the row     |
|              | index of each non-zero element of A matrix; The       |
|              | length is at least ``rows_end[rows-1] – ind``;When    |
|              | the input array is addressed based on 0, the value of |
|              | ind is 0;When addressed based on 1, the value of ind  |
|              | is 1.                                                 |
+--------------+-------------------------------------------------------+
| values       | Store the value of the non-zero element in the matrix |
|              | A, the length is equal to the length of ``row_indx``  |
+--------------+-------------------------------------------------------+

alphasparse_create_bsr
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_create_bsr( 
       alphasparse_matrix_t *A, 
       const alphasparse_index_base_t indexing, 
       const alphasparse_layout_t block_layout, 
       const ALPHA_INT rows, 
       const ALPHA_INT cols, 
       const ALPHA_INT block_size, 
       ALPHA_INT *rows_start, 
       ALPHA_INT *rows_end, 
       ALPHA_INT *col_indx, 
       ALPHA_Number * values)

The ``alphasparse_?_create_bsr`` is used to create a sparse matrix in
``BSR`` matrix format. The matrix size is ``m*k`` and is stored in
variable ``A``. “``?``” indicates the data format. It corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, and ``c`` corresponds to float complex, that is
single-precision complex number, ``z``\ corresponds to double complex,
that is double-precision complex number, other parameters are shown
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| A            | BSR format                                            |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, the index starts with                   |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, the index starts with 1                   |
+--------------+-------------------------------------------------------+
| block_layout | Describe the storage mode of non-zero elements in the |
|              | sparse matrix block, with the following               |
|              | options:\ ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``, Row      |
|              | major design\ ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``,   |
|              | Column major design                                   |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of non-zero block of matrix ``A``      |
+--------------+-------------------------------------------------------+
| cols         | The number of columns in the non-zero block of matrix |
|              | ``A``                                                 |
+--------------+-------------------------------------------------------+
| block_size   | The length of the non-zero element block of the       |
|              | sparse matrix, the size of each non-zero element      |
|              | block is ``block_size * block_size``                  |
+--------------+-------------------------------------------------------+
| rows_start   | The length is at least m, contains the index of each  |
|              | non-zero block row of the                             |
|              | matrix,\ ``rows_start[i] – ind`` is the starting      |
|              | index of the i-th block row in values and             |
|              | ``col_indx``; when the input array is addressed based |
|              | on 0, the value of ind is 0,when addressed based on   |
|              | 1, the value of ind is 1.                             |
+--------------+-------------------------------------------------------+
| rows_end     | The length is at least m, contains the index of each  |
|              | non-zero block row of the matrix,                     |
|              | \ ``rows_end[i] – ind`` is the end position of the    |
|              | i-th block row in values and ``col_indx``;when the    |
|              | input array is based on 0 addressing, the value of    |
|              | ind is 0, and when addressing based on 1, the value   |
|              | of ind is 1                                           |
+--------------+-------------------------------------------------------+
| col_indx     | When addressing based on 1, the array contains the    |
|              | row index of each non-zero block of matrix ``A`` + 1, |
|              | when addressing based on 0, the array contains the    |
|              | row index of each non-zero block of matrix ``A``; The |
|              | length is at least ``rows_end[rows-1] – ind``,When    |
|              | the input array is addressed based on 0, the value of |
|              | ind is 0, When addressed based on 1, the value of ind |
|              | is 1                                                  |
+--------------+-------------------------------------------------------+
| values       | store the value of non-zero elements in ``A``, the    |
|              | length equals ``col_indx*block_size*block_size``      |
|              | quite                                                 |
+--------------+-------------------------------------------------------+

alphasparse_convert_csr
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_convert_csr( 
       const alphasparse_matrix_t source,
       const alphasparse_operation_t operation, 
       alphasparse_matrix_t *dest)

The ``alphasparse_convert_csr`` is used to convert the data structure of
other sparse matrix format into the data structure of CSR matrix format,
which is stored in dest. The parameter explanation is shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | Source matrix                                         |
+--------------+-------------------------------------------------------+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose, ``op(A) = AT``                             |
+--------------+-------------------------------------------------------+
| dest         | Matrix in CSR format                                  |
+--------------+-------------------------------------------------------+

alphasparse_convert_csc
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_convert_csc( 
       const alphasparse_matrix_t source, 
       const alphasparse_operation_t operation, 
       alphasparse_matrix_t *dest)

The ``alphasparse_convert_csc`` converts the data structure of other
sparse matrix format to the data structure of CSC matrix format, which
is stored in dest. The parameter explanation is shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | Source matrix                                         |
+--------------+-------------------------------------------------------+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose, ``op(A) = AT``                             |
+--------------+-------------------------------------------------------+
| dest         | Matrix in CSC format                                  |
+--------------+-------------------------------------------------------+

alphasparse_convert_sky
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_convert_sky( 
       const alphasparse_matrix_t source, 
       const alphasparse_operation_t operation, 
       alphasparse_matrix_t *dest)

The ``alphasparse_convert_sky`` converts the data structure of other
sparse matrix format to the data structure of SKY matrix format, which
is stored in dest. The parameter explanation is shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | Source matrix                                         |
+--------------+-------------------------------------------------------+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose, ``op(A) = AT``                             |
+--------------+-------------------------------------------------------+
| dest         | Matrix in SKY format                                  |
+--------------+-------------------------------------------------------+

alphasparse_convert_dia
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_convert_dia( 
       const alphasparse_matrix_t source, 
       const alphasparse_operation_t operation, 
       alphasparse_matrix_t *dest)

The ``alphasparse_convert_dia`` converts the data structure of other
sparse matrix format to the data structure of DIA matrix format, which
is stored in dest. The parameter explanation is shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | Source matrix                                         |
+--------------+-------------------------------------------------------+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose, ``op(A) = AT``                             |
+--------------+-------------------------------------------------------+
| dest         | Matrix in DIA format                                  |
+--------------+-------------------------------------------------------+

alphasparse_convert_bsr
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_convert_bsr( 
       const alphasparse_matrix_t source, 
       const alphasparse_operation_t operation, 
       alphasparse_matrix_t *dest)

The ``alphasparse_convert_bsr`` converts the data structure of other
sparse matrix format to the data structure of BSR matrix format, which
is stored in dest. The parameter explanation is shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | Source matrix                                         |
+--------------+-------------------------------------------------------+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose, ``op(A) = AT``                             |
+--------------+-------------------------------------------------------+
| dest         | Matrix in BSR format                                  |
+--------------+-------------------------------------------------------+

.. _alphasparse_convert_csc-1:

alphasparse_convert_csc
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_export_csc( 
       alphasparse_matrix_t source, 
       alphasparse_index_base_t *indexing, 
       ALPHA_INT *rows, 
       ALPHA_INT *cols, 
       ALPHA_INT **cols_start, 
       ALPHA_INT **cols_end, 
       ALPHA_INT **row_indx, 
       ALPHA_Number ** values)

The ``alphasparse_?_export_csc`` converts ``m*k`` CSC to a multiple data
variables CSC. “``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is
single-precision complex number, and ``z`` corresponds to double
complex, which is double-precision complex number. Other parameters are
shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | CSC format                                            |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, the index starts with                   |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, the index starts with 1                   |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of matrix ``A``                        |
+--------------+-------------------------------------------------------+
| cols         | Number of columns of matrix ``A``                     |
+--------------+-------------------------------------------------------+
| cols_start   | The length is at least m, contains the index of each  |
|              | column of the matrix,\ ``cols_start[i] – ind`` is the |
|              | starting index of the ``i-th`` column in values and   |
|              | ``row_indx``;when the input array is addressed based  |
|              | on 0, the value of ind is 0, when addressed based on  |
|              | 1, the value of ind is 1.                             |
+--------------+-------------------------------------------------------+
| cols_end     | The length is at least m, contains the index of each  |
|              | column of the matrix,\ ``cols_end[i] – ind`` is the   |
|              | end position of the ``i-th`` column in values and     |
|              | ``row_indx``; when the input array is addressed based |
|              | on 0, the value of ind is 0, when addressed based on  |
|              | 1, the value of ind is 1                              |
+--------------+-------------------------------------------------------+
| row_indx     | When addressing based on 1, the array contains the    |
|              | row index of each non-zero element of ``A`` +1. When  |
|              | addressing based on 0, the array contains the row     |
|              | index of each non-zero element of ``A``; the length   |
|              | is at least ``cols_end[cols-1] – ind``,When the input |
|              | array is addressed based on 0, the value of ``ind``   |
|              | is 0, when addressed based on 1, the value of ind is  |
|              | 1                                                     |
+--------------+-------------------------------------------------------+
| values       | Store the value of non-zero element in the matrix A,  |
|              | the length is equivalent to the length of             |
|              | ``row_indx``                                          |
+--------------+-------------------------------------------------------+

alphasparse_export_csr
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_export_csr(
       alphasparse_matrix_t source, 
       const alphasparse_index_base_t *indexing, 
       const ALPHA_INT *rows, 
       const ALPHA_INT *cols, 
       ALPHA_INT **rows_start, 
       ALPHA_INT **rows_end, 
       ALPHA_INT **col_indx, 
       ALPHA_Number ** values)

The ``alphasparse_?_export_csr`` converts ``m*k`` CSR to a multiple data
variables CSR. “``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is
single-precision complex number, and ``z`` corresponds to double
complex, which is double-precision complex number. Other parameters are
shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | CSR Format                                            |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, the index starts with                   |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, the index starts with 1                   |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of matrix ``A``                        |
+--------------+-------------------------------------------------------+
| cols         | Number of columns of matrix ``A``                     |
+--------------+-------------------------------------------------------+
| rows_start   | The length is at least m, contains the index of each  |
|              | rows of the matrix,\ ``rows_start[i] – ind`` is the   |
|              | starting index of the ``i-th`` rows in values and     |
|              | ``col_indx``;when the input array is addressed based  |
|              | on 0, the value of ind is 0, when addressed based on  |
|              | 1, the value of ind is 1.                             |
+--------------+-------------------------------------------------------+
| rows_end     | The length is at least m, contains the index of each  |
|              | rows of the matrix,\ ``row_end[i] – ind`` is the end  |
|              | position of the ``i-th`` rows in values and           |
|              | ``col_indx``; when the input array is addressed based |
|              | on 0, the value of ind is 0, when addressed based on  |
|              | 1, the value of ind is 1                              |
+--------------+-------------------------------------------------------+
| col_indx     | When addressing based on 1, the array contains the    |
|              | column index of each non-zero element of ``A`` +1.    |
|              | When addressing based on 0, the array contains the    |
|              | column index of each non-zero element of ``A``; the   |
|              | length is at least ``cols_end[cols-1] – ind``,When    |
|              | the input array is addressed based on 0, the value of |
|              | ``ind`` is 0, when addressed based on 1, the value of |
|              | ind is 1                                              |
+--------------+-------------------------------------------------------+
| values       | Store the value of non-zero element in the matrix A,  |
|              | the length is equivalent to the length of row_indx    |
+--------------+-------------------------------------------------------+

alphasparse_export_bsr
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_export_bsr( 
       alphasparse_matrix_t source, 
       alphasparse_index_base_t *indexing, 
       alphasparse_layout_t *block_layout, 
       ALPHA_INT *rows, 
       ALPHA_INT *cols, 
       ALPHA_INT *block_size, 
       ALPHA_INT **rows_start, 
       ALPHA_INT **rows_end, 
       ALPHA_INT **col_indx, 
       ALPHA_Number ** values)

The ``alphasparse_?_export_bsr`` converts ``m*k`` BSR to a multiple data
variables BSR. “``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is
single-precision complex number, and ``z`` corresponds to double
complex, which is double-precision complex number. Other parameters are
shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | BSR format                                            |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, the index starts with                   |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, the index starts with 1                   |
+--------------+-------------------------------------------------------+
| block_layout | Describe the storage mode of non-zero elements in the |
|              | sparse matrix block, with the following               |
|              | options:\ ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``,Row major |
|              | design\ ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``,Column   |
|              | major design                                          |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of non-zero block of matrix ``A``      |
+--------------+-------------------------------------------------------+
| cols         | The number of columns in the non-zero block of matrix |
|              | ``A``                                                 |
+--------------+-------------------------------------------------------+
| block_size   | length of non-zero element block of matrix, size of   |
|              | each non-zero block is ``block_size *  block_size``   |
+--------------+-------------------------------------------------------+
| rows_start   | The length is at least m, contains the index of each  |
|              | non-zero block row of the                             |
|              | matrix,\ ``rows_start[i] – indIt`` is the starting    |
|              | index of the ``i-th`` block row in ``values`` and     |
|              | ``col_indx``; when the input array is addressed based |
|              | on 0, the value of ind is 0,when addressed based on   |
|              | 1, the value of ind is 1.                             |
+--------------+-------------------------------------------------------+
| rows_end     | The length is at least m, contains the index of each  |
|              | non-zero block row of the matrix,                     |
|              | \ ``rows_end[i] – ind`` It is the end position of the |
|              | i-th block row in ``values`` and ``col_indx``; when   |
|              | the input array is based on 0 addressing, the value   |
|              | of ind is 0, when addressing based on 1, the value of |
|              | ind is 1                                              |
+--------------+-------------------------------------------------------+
| col_indx     | When addressing based on 1, the array contains the    |
|              | row index of each non-zero block of matrix ``A`` + 1, |
|              | when addressing based on 0, the array contains the    |
|              | row index of each non-zero block of matrix ``A``; the |
|              | length is at least ``rows_end[rows-1] – ind``,When    |
|              | the input array is addressed based on 0, the value of |
|              | ind is 0,When addressed based on 1, the value of ind  |
|              | is 1                                                  |
+--------------+-------------------------------------------------------+
| values       | Store the value of non-zero elements in matrix A, the |
|              | length is ``col_indx*block_size*block_size``\ quite   |
+--------------+-------------------------------------------------------+

alphasparse_export_coo
^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_export_coo( 
       alphasparse_matrix_t source, 
       alphasparse_index_base_t *indexing, 
       ALPHA_INT *rows, 
       ALPHA_INT *cols, 
       ALPHA_INT **row_indx, 
       ALPHA_INT **col_indx, 
       ALPHA_Number * values, 
       ALPHA_INT *nnz)

The ``alphasparse_?_export_coo`` converts ``m*k`` COO to a multiple data
variables COO. “``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is
single-precision complex number, and ``z`` corresponds to double
complex, which is double-precision complex number. Other parameters are
shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| source       | COO formatMatrixsourcedata structure                  |
+--------------+-------------------------------------------------------+
| indexing     | Indicates the addressing mode of the input            |
|              | array,There are the following                         |
|              | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based on  |
|              | 0 addressing, the index starts with                   |
|              | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1        |
|              | addressing, the index starts with 1                   |
+--------------+-------------------------------------------------------+
| rows         | Number of rows of matrix ``A``                        |
+--------------+-------------------------------------------------------+
| cols         | Number of columns of matrix ``A``                     |
+--------------+-------------------------------------------------------+
| row_indx     | The row coordinate index of each non-zero element,    |
|              | the length is ``nnz``                                 |
+--------------+-------------------------------------------------------+
| col_indx     | The column coordinate index of each non-zero element, |
|              | the length is ``nnz``                                 |
+--------------+-------------------------------------------------------+
| values       | Store the values of non-zero elements in matrix ``A`` |
|              | in any order, with a length of ``nnz``                |
+--------------+-------------------------------------------------------+
| nnz          | The number of non-zero elements of matrix ``A``       |
+--------------+-------------------------------------------------------+

alphasparse_destroy
^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_destroy(
       alphasparse_matrix_t A)

The ``alphasparse_destroy``, The function performs the operation of
releasing the memory space occupied by the sparse matrix data structure.
The only input parameter required is the to be released ``A`` of the
sparse matrix.

CPU backend
-----------

Multiplying sparse matrix and dense vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

   alphasparse_status_t alphasparse_?_mv( 
       const alphasparse_operation_t operation,
       const ALPHA_Number alpha, 
       const alphasparse_matrix_t A, 
       const struct AlphaSparse_matrix_descr descr, 
       const ALPHA_Number *x, 
       const ALPHA_Number beta, 
       ALPHA_Number *y) 

The ``alphasparse_?_mv`` function performs the operation of multiplying
a sparse matrix and a dense vector:

.. math:: y := alpha \times op(A) \times x + beta \times y

Alpha and beta are scalar values, ``A`` is a sparse matrix with ``k``
rows and ``m`` columns, ``x`` and ``y`` are vectors. “``?``” indicates
the data format, which corresponds to the ``ALPHA_Number`` in the
interface, ``s`` corresponds to float, ``d`` corresponds to double, and
``c`` corresponds to float complex, which is a single-precision complex
number, and ``z`` corresponds to a double complex, which is a
double-precision complex number. This function stores the output result
in the vector ``y``. The input parameters of the function are shown
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose,                                            |
|              | ``op(A) = AT                                          |
|              | ``\ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|              | ConjugationTranspose, ``op(A) = AH``                  |
+--------------+-------------------------------------------------------+
| alpha        | Scalar value ``alpha``                                |
+--------------+-------------------------------------------------------+
| A            | Data structure of sparse matrix                       |
+--------------+-------------------------------------------------------+
| descr        | This structure describes a sparse matrix with special |
|              | structural attributes, and has three members:         |
|              | \ ``type``, ``mode``, and ``diag``.\ ``type``         |
|              | indicates the type of                                 |
|              | matrix:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,        |
|              | General                                               |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC``,       |
|              | Symmetric                                             |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_ HERMITIAN``,      |
|              | Hermit                                                |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``,      |
|              | Triangular                                            |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,        |
|              | Diagonal                                              |
|              | m                                                     |
|              | atrix\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|              | Block Triangular matrix(Only in sparse matrix format  |
|              | BSR                                                   |
|              | )\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``,Block, |
|              | Diagonal matrix(Only in sparse matrix format          |
|              | BSR)\ ``Mode`` specifies the triangular part to be    |
|              | processed for symmetric matrix and triangular         |
|              | matrix\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, processing  |
|              | the lower triangular of the                           |
|              | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, processing  |
|              | the upper triangular of the matrix\ ``Diag``          |
|              | indicates whether the non-zero elements of the        |
|              | diagonal in the non-general matrix are equal to       |
|              | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all diagonal  |
|              | elements are equal to 1\ ``ALPHA_SPARSE_DIAG_UNIT``,  |
|              | the diagonal elements are all equal to 1              |
+--------------+-------------------------------------------------------+
| x            | Dense vector ``x``, stored as an array, if no         |
|              | transpose operation is performed on matrix ``A``, the |
|              | length is at least the number of columns of matrix    |
|              | ``A``                                                 |
+--------------+-------------------------------------------------------+
| beta         | Scalar value ``beta``                                 |
+--------------+-------------------------------------------------------+
| y            | Dense vector ``y``, stored as an array, if no         |
|              | transpose operation is performed on matrix ``A``, the |
|              | length is at least the number of rows of matrix ``A`` |
+--------------+-------------------------------------------------------+

Multiplying sparse matrix and dense matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

   alphasparse_status_t alphasparse_?_mm(
       const alphasparse_operation_t operation, 
       const ALPHA_Number alpha, 
       const alphasparse_matrix_t A, 
       const struct AlphaSparse_matrix_descr descr, 
       const alphasparse_layout_t layout, 
       const ALPHA_Number *x, 
       const ALPHA_INT columns, 
       const ALPHA_INT ldx, 
       const ALPHA_Number beta, ALPHA_Number *y,  
       const ALPHA_INT ldy)

The ``alphasparse_?_mm`` function performs the operation of multiplying
a sparse matrix and a dense matrix:

.. math:: y := alpha \times op(A) \times x + beta \times y

``Alpha`` and ``beta`` are scalar values, ``A`` is a sparse matrix,
``x`` and ``y`` are dense matrices, “``?``” indicates the data format,
which corresponds to the ``ALPHA_Number`` in the interface, ``s``
corresponds to float, ``d`` corresponds to double, and ``c`` corresponds
to float complex, which is a single-precision complex number, ``z``
corresponds to double complex, which is double-precision complex number,
this function stores the result in matrix y. The input parameters of the
function are shown as below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose,                                            |
|              | ``op(A) = AT                                          |
|              | ``\ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|              | Conjugation Transpose, ``op(A) = AH``                 |
+--------------+-------------------------------------------------------+
| alpha        | ``Scalar`` value alpha                                |
+--------------+-------------------------------------------------------+
| A            | Data structure of sparse matrix                       |
+--------------+-------------------------------------------------------+
| descr        | This structure describes a sparse matrix with special |
|              | structural attributes, and has three members:         |
|              | \ ``type``, ``mode``, and ``diag``:\ ``type``         |
|              | indicates the type of                                 |
|              | matrix:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,        |
|              | general                                               |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC``,       |
|              | symmetric                                             |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_ HERMITIAN``,      |
|              | Hermit                                                |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``,      |
|              | triangular                                            |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,        |
|              | diagonal                                              |
|              | m                                                     |
|              | atrix\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|              | Block Triangular matrix (Only in sparse matrix format |
|              | BS                                                    |
|              | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``,Block |
|              | Diagonal matrix(Only in sparse matrix format          |
|              | BSR)\ ``Mode`` specifies the triangular part to be    |
|              | processed for symmetric matrix and triangular         |
|              | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, processing |
|              | the lower part of the                                 |
|              | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, processing  |
|              | the upper part of the matrix\ ``Diag`` indicates      |
|              | whether the non-zero elements of the diagonal in the  |
|              | non-general matrix are equal to                       |
|              | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all diagonal  |
|              | elements are equal to 1\ ``ALPHA_SPARSE_DIAG_UNIT``,  |
|              | the diagonal elements are all equal to 1              |
+--------------+-------------------------------------------------------+
| layout       | Describe the storage mode of dense                    |
|              | matrix:\ ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``, row major |
|              | design\ ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``, column  |
|              | major design                                          |
+--------------+-------------------------------------------------------+
| x            | Dense matrix ``x``, stored as an array, with a length |
|              | of at least rows*cols                                 |
+--------------+-------------------------------------------------------+
| columns      | Number of columns of dense matrix ``y``               |
+--------------+-------------------------------------------------------+
| ldx          | Specify the size of the main dimension of the matrix  |
|              | ``x`` when it is actually stored                      |
+--------------+-------------------------------------------------------+
| beta         | Scalar value ``beta``                                 |
+--------------+-------------------------------------------------------+
| y            | Dense matrix ``y``, stored as an array, with a length |
|              | of at least rows*cols, where                          |
+--------------+-------------------------------------------------------+
| ldy          | Specify the size of the main dimension of the matrix  |
|              | ``y`` when it is actually stored                      |
+--------------+-------------------------------------------------------+

For param denes matrix ``x``, data layouts is showed below:

+----------------------------+---------+-------------------------------+
|                            | Column  | Row major design              |
|                            | major   |                               |
|                            | design  |                               |
+============================+=========+===============================+
| The rows value (the number | ``ldx`` | When ``op(A) = A``, it is the |
| of rows in matrix ``x``)   |         | number of columns of          |
| is                         |         | ``A``\ When ``op(A) = AT``,   |
|                            |         | it is the number of rows of   |
|                            |         | ``A``                         |
+----------------------------+---------+-------------------------------+
| The cols value (the number | columns | ``ldx``                       |
| of columns of matrix       |         |                               |
| ``x``) is                  |         |                               |
+----------------------------+---------+-------------------------------+

For param denes matrix ``y``, data layouts is shown below:

+----------------------------+---------+-------------------------------+
|                            | Column  | Row major design              |
|                            | major   |                               |
|                            | design  |                               |
+============================+=========+===============================+
| The rows value (the number | ``ldy`` | When ``op(A) = A``, it is the |
| of rows in matrix ``y``)   |         | number of columns of          |
| is                         |         | ``A``\ When ``op(A) = AT``,   |
|                            |         | it is the number of rows of   |
|                            |         | ``A``                         |
+----------------------------+---------+-------------------------------+
| The cols value (the number | columns | ``ldy``                       |
| of columns of matrix       |         |                               |
| ``y``) is                  |         |                               |
+----------------------------+---------+-------------------------------+

Sparse matrix and sparse matrix multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functions are divided into two categories according to the different
output results:

3.1 alphasparse_spmmd
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_spmmd( 
       const alphasparse_operation_t operation, 
       const alphasparse_matrix_t A, 
       const alphasparse_matrix_t B, 
       const alphasparse_layout_t layout, ALPHA_Number *C, 
       const ALPHA_INT ldc) 

The ``alphasparse_?_spmmd`` performs the operation of multiplying a
sparse matrix and a **dense** matrix:

.. math:: C := op(A) \times B

``A`` is sparse matrices, ``B`` is a dense matrix and ``C`` is a dense
matrix which also stores the output result of the function. “``?``”
indicates the data format, which corresponds to the ``ALPHA_Number`` in
the interface. ``s`` corresponds to float, ``d`` corresponds to double,
``c`` corresponds to float complex, which is a single-precision complex
number, and ``z`` corresponds to double complex, which is double
precision. The input parameters of the function are shown in below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | no transposition,                                     |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose,                                            |
|              | ``op(A) = AT                                          |
|              | ``\ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|              | Conjugation Transpose, ``op(A) = AH``                 |
+--------------+-------------------------------------------------------+
| A            | Data structure of sparse matrix                       |
+--------------+-------------------------------------------------------+
| B            | Data structure of dense matrix                        |
+--------------+-------------------------------------------------------+
| layout       | Describe the storage mode of dense matrix:            |
|              | ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``, row major design   |
|              | ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``, column major    |
|              | design                                                |
+--------------+-------------------------------------------------------+
| C            | Dense matrix ``C``                                    |
+--------------+-------------------------------------------------------+
| ldc          | Specify the size of the main dimension of the matrix  |
|              | ``C`` when it is actually stored                      |
+--------------+-------------------------------------------------------+

3.2 alphasparse_spmm
^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_spmm( 
       const alphasparse_operation_t operation, 
       const alphasparse_matrix_t A, 
       const alphasparse_matrix_t B, 
       alphasparse_matrix_t *C) 

The ``alphasparse_?_spmm`` performs the operation of multiplying a
sparse matrix and a **sparse** matrix:

.. math:: C := op(A) \times B

``A`` and ``B`` are sparse matrices, ``C`` is a sparse matrix, and the
output result of the function is stored at the same time. “``?``”
indicates the data format, which corresponds to the ``ALPHA_Number`` in
the interface. ``s`` corresponds to float, ``d`` corresponds to double,
and ``c`` corresponds to float complex, namely Single-precision complex
number, ``z`` corresponds to double complex, a double-precision complex
number. The input parameters of the function are shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | non-transposed,                                       |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose,                                            |
|              | ``op(A) = AT                                          |
|              | ``\ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|              | Conjugation Transpose, ``op(A) = AH``                 |
+--------------+-------------------------------------------------------+
| A            | Data structure of sparse matrix                       |
+--------------+-------------------------------------------------------+
| B            | Another sparse matrix data structure                  |
+--------------+-------------------------------------------------------+
| C            | Data structure of sparse matrix C                     |
+--------------+-------------------------------------------------------+

Solving linear equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.1 alphasparse_trsv
^^^^^^^^^^^^^^^^^^^^

Equations for multiplying a sparse matrix and a dense vector:

.. code:: cpp

   alphasparse_status_t alphasparse_?_trsv( 
       const alphasparse_operation_t operation, 
       const ALPHA_Number alpha, 
       const alphasparse_matrix_t A, 
       const struct AlphaSparse_matrix_descr descr, 
       const ALPHA_Number *x,ALPHA_Number *y) 

The ``alphasparse_?_trsv`` function performs the operation of solving
the equations of the matrix:

.. math:: op(A)\times y = alpha \times x

``Alpha`` is a scalar value, and ``A`` is a triangular sparse matrix. If
A is not a triangular matrix, only the needed part of the triangular
matrix is processed. ``x`` and ``y`` are vectors, and “``?``” indicates
the data format, which corresponds to the ``ALPHA_Number`` in the
interface. ``s`` corresponds to float, ``d`` corresponds to double,
``c`` corresponds to float complex, which is a single-precision complex
number, and ``z`` corresponds to double complex, which is a
double-precision complex number. This function stores the output result
in the vector ``y``. The input parameter is shown below.

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | non-transposed,                                       |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose,                                            |
|              | ``op(A) = AT                                          |
|              | ``\ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|              | Conjugation Transpose, ``op(A) = AH``                 |
+--------------+-------------------------------------------------------+
| alpha        | Scalar value ``alpha``                                |
+--------------+-------------------------------------------------------+
| A            | Data structure of sparse matrix                       |
+--------------+-------------------------------------------------------+
| descr        | This structure describes a sparse matrix with special |
|              | structural attributes, and has three                  |
|              | members:\ ``type``, ``mode``, and ``diag``. The       |
|              | ``type`` member indicates the matrix                  |
|              | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``, general  |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,        |
|              | diagonal                                              |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``,      |
|              | triangular                                            |
|              | m                                                     |
|              | atrix\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|              | Block Triangular matrix (Only in sparse matrix format |
|              | BSR)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``,    |
|              | Block Diagonal matrix (Only in sparse matrix BSR      |
|              | format)The ``mode`` member indicates the triangular   |
|              | characteristics of the                                |
|              | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower      |
|              | triangular matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,  |
|              | upper triangular matrix\ ``Diag`` indicates whether   |
|              | the non-zero elements of the diagonal matrix are      |
|              | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all  |
|              | diagonal elements are equal to                        |
|              | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal elements  |
|              | are all equal to 1                                    |
+--------------+-------------------------------------------------------+
| x            | Dense vector ``x``                                    |
+--------------+-------------------------------------------------------+
| beta         | Scalar value ``beta``                                 |
+--------------+-------------------------------------------------------+
| y            | Dense vector ``y``                                    |
+--------------+-------------------------------------------------------+

4.2 alphasparse_trsm
^^^^^^^^^^^^^^^^^^^^

A system of equations for multiplying a sparse matrix and a dense
matrix:

.. code:: cpp

   alphasparse_status_t alphasparse_?_trsm( 
       const alphasparse_operation_t operation, 
       const ALPHA_Number alpha, 
       const alphasparse_matrix_t A, 
       const struct AlphaSparse_matrix_descr descr, 
       const alphasparse_layout_t layout, 
       const ALPHA_Number *x, 
       const ALPHA_INT columns, 
       const ALPHA_INT ldx, 
       ALPHA_Number *y, 
       const ALPHA_INT ldy)

The ``alphasparse_?_trsm`` function performs the operation of solving
the equations of the matrix:

.. math:: y := alpha\times inv(op(A))\times x

``Alpha`` is a scalar value, and ``inv(op(A))`` is the inverse matrix of
the triangular sparse matrix. If ``A`` is not a triangular matrix, only
the required part of the triangular matrix will be processed. ``x`` and
``y`` are vectors, and “``?``” indicates the data format, which
corresponds to the ``ALPHA_Number`` in the interface. ``s`` corresponds
to float, ``d`` corresponds to double, ``c`` corresponds to float
complex, which is a single-precision complex number, and ``z``
corresponds to double complex, which is a double-precision complex
number. The function stores the output result in the vector ``y``. The
input parameters of the function are shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| operation    | For specific operations on the input matrix, there    |
|              | are the following                                     |
|              | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``,   |
|              | non-transposed,                                       |
|              | `                                                     |
|              | `op(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|              | transpose,                                            |
|              | ``op(A) = AT                                          |
|              | ``\ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|              | Conjugation Transpose, ``op(A) = AH``                 |
+--------------+-------------------------------------------------------+
| alpha        | Scalar value ``alpha``                                |
+--------------+-------------------------------------------------------+
| A            | Data structure of sparse matrix                       |
+--------------+-------------------------------------------------------+
| descr        | This structure describes a sparse matrix with special |
|              | structural attributes, and has three                  |
|              | members:\ ``type``, ``mode``, and ``diag``. The       |
|              | ``type`` member indicates the matrix                  |
|              | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``, general  |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,        |
|              | diagonal                                              |
|              | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``,      |
|              | triangular                                            |
|              | m                                                     |
|              | atrix\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|              | Block Triangular matrix (Only in sparse matrix format |
|              | BSR)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``,    |
|              | Block Diagonal matrix (Only in sparse matrix BSR      |
|              | format)The ``mode`` member indicates the triangular   |
|              | characteristics of the                                |
|              | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower      |
|              | triangular matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,  |
|              | upper triangular matrix\ ``Diag`` indicates whether   |
|              | the non-zero elements of the diagonal matrix are      |
|              | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all  |
|              | diagonal elements are equal to                        |
|              | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal elements  |
|              | are all equal to 1                                    |
+--------------+-------------------------------------------------------+
| layout       | Describe the storage mode of dense                    |
|              | matrix:\ ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``, row major |
|              | design\ ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``, column  |
|              | major design                                          |
+--------------+-------------------------------------------------------+
| x            | ``x``, input as a parameter, is stored in an array,   |
|              | and the length is at least ``rows*cols``              |
+--------------+-------------------------------------------------------+
| columns      | Number of columns of dense matrix ``y``               |
+--------------+-------------------------------------------------------+
| ldx          | Specify the size of the main dimension of the matrix  |
|              | x when it is actually stored                          |
+--------------+-------------------------------------------------------+
| beta         | Scalar value ``beta``                                 |
+--------------+-------------------------------------------------------+
| y            | Dense matrix ``y``, stored as an array, with a length |
|              | of at least ``rows*cols``,                            |
+--------------+-------------------------------------------------------+
| ldy          | Specify the size of the main dimension of the matrix  |
|              | y when it is actually stored                          |
+--------------+-------------------------------------------------------+

For param denes matrix ``x``, data layouts is showed below:

+----------------------------+---------+-------------------------------+
|                            | Column  | Row major design              |
|                            | major   |                               |
|                            | design  |                               |
+============================+=========+===============================+
| The rows value (the number | ``ldx`` | When ``op(A) = A``, it is the |
| of rows in matrix ``x``)   |         | number of columns of          |
| is                         |         | ``A``\ When ``op(A) = AT``,   |
|                            |         | it is the number of rows of   |
|                            |         | ``A``                         |
+----------------------------+---------+-------------------------------+
| The cols value (the number | columns | ``ldx``                       |
| of columns of matrix       |         |                               |
| ``x``) is                  |         |                               |
+----------------------------+---------+-------------------------------+

For param denes matrix ``y``, data layouts is shown below:

+----------------------------+---------+-------------------------------+
|                            | Column  | Row major design              |
|                            | major   |                               |
|                            | design  |                               |
+============================+=========+===============================+
| The rows value (the number | ``ldy`` | When ``op(A) = A``, it is the |
| of rows in matrix ``y``)   |         | number of columns of          |
| is                         |         | ``A``\ When ``op(A) = AT``,   |
|                            |         | it is the number of rows of   |
|                            |         | ``A``                         |
+----------------------------+---------+-------------------------------+
| The cols value (the number | columns | ``ldy``                       |
| of columns of matrix       |         |                               |
| ``y``) is                  |         |                               |
+----------------------------+---------+-------------------------------+

level1 Vector operation
~~~~~~~~~~~~~~~~~~~~~~~~~~

alphasparse_axpy
^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_axpy ( 
       const ALPHA_INT nz, 
       const ALPHA_Number a, 
       const ALPHA_Number *x, 
       const ALPHA_INT *indx, 
       ALPHA_Number * y)

The ``alphasparse_?_ axpy`` executes the operation of adding multiple
scalar values of a compressed vector to the full storage vector:

.. math:: y := a\times x + y

``a`` is a scalar value, ``x`` is a sparse vector in compressed format,
``y`` is a fully stored vector. “``?``” indicates the data format, which
corresponds to the ``ALPHA_Number`` in the interface. ``s`` corresponds
to float, ``d`` corresponds to double, ``c`` corresponds to float
complex, which is a single-precision complex number, and ``z``
corresponds to double complex, which is a double-precision complex
number. This function stores the output result in In the vector ``y``.
The input parameters of the function are shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors ``x`` and indx          |
+--------------+-------------------------------------------------------+
| a            | Scalar value ``a``                                    |
+--------------+-------------------------------------------------------+
| x            | Store as an array, The length is at least ``nz``      |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of the vector ``x``, Store as |
|              | an array, The length is at least ``nz``               |
+--------------+-------------------------------------------------------+
| y            | Store as an array, The length is at least             |
|              | ``max(indx[i])``                                      |
+--------------+-------------------------------------------------------+

alphasparse_gthr
^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_gthr ( 
       const ALPHA_INT nz, 
       const ALPHA_Number * y, 
       ALPHA_Number *x, 
       const ALPHA_INT *indx)

The ``alphasparse_?_ gthr`` executes by index of gathering the elements
of a full storage vector into a compressed vector format:

.. math:: x[i] = y[indx[i]], i=0,1,... ,nz-1

Here ``x`` is a sparse vector in compressed format, ``y`` is a fully
stored vector.“``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This function
stores the output result in vector ``x``. The input parameters of the
function are shown in below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors ``x`` and ``indx``      |
+--------------+-------------------------------------------------------+
| y            | Store as an array, The length is at least             |
|              | ``max(indx[i])``                                      |
+--------------+-------------------------------------------------------+
| x            | Store as an array, The length is at least ``nz``      |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of the vector ``x``, store as |
|              | an array, The length is at least ``nz``               |
+--------------+-------------------------------------------------------+

alphasparse_gthrz
^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_gthrz ( 
       const ALPHA_INT nz, 
       ALPHA_Number * y, 
       ALPHA_Number *x, 
       const ALPHA_INT *indx)

The ``alphasparse_?_ gthrz`` executes by index of gathering the elements
of a full storage vector into the compressed vector format, and zeroing
the elements at the corresponding positions in the original vector:

.. math:: x[i] = y[indx[i]], y[indx[i]] = 0, i=0,1,... ,nz-1

Here ``x`` is a sparse vector in compressed format, ``y`` is a fully
stored vector. “``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This output result
is the updated compression vector ``x`` and updated ``y``. The input
parameters of the function are shown in below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors ``x`` and ``indx``      |
+--------------+-------------------------------------------------------+
| y            | Store as an array, The length is at least             |
|              | ``max(indx[i])``                                      |
+--------------+-------------------------------------------------------+
| x            | Store as an array, The length is at least ``nz``      |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of the vector ``x``, store as |
|              | an array, The length is at least ``nz``               |
+--------------+-------------------------------------------------------+

alphasparse_rot
^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_rot ( 
       const ALPHA_INT nz, 
       ALPHA_Number *x, 
       const ALPHA_INT *indx, 
       ALPHA_Number * y,
       const ALPHA_Number c, 
       const ALPHA_Number s)

The ``alphasparse_?_ rot``, performs the conversion operation of two
real number vectors:

.. math:: x[i] = c\times x[i] + s\times y[indx[i]]

.. math:: y[indx[i]] = c\times y[indx[i]]- s\times x[i]

Here ``x`` is a sparse vector in compressed format, ``y`` is a fully
stored vector, The value of indx must be unique. “``?``” indicates the
data format, which corresponds to the ``ALPHA_Number`` in the interface.
``s`` corresponds to float and ``d`` corresponds to double. This output
is updated vector ``x`` and ``y``. The input parameters of the function
are shown in below:

+------------------+--------------------------------------------------+
| Input parameters | Description                                      |
+==================+==================================================+
| nz               | Number of elements in vectors ``x`` and ``indx`` |
+------------------+--------------------------------------------------+
| x                | Store as an array, The length is at least ``nz`` |
+------------------+--------------------------------------------------+
| indx             | Index of the vector ``x``, saved as an array,    |
|                  | length is at least ``nz``                        |
+------------------+--------------------------------------------------+
| y                | Store as an array, the length is at least        |
|                  | ``max(indx[i])``                                 |
+------------------+--------------------------------------------------+
| c                | Scalar value                                     |
+------------------+--------------------------------------------------+
| s                | Scalar value                                     |
+------------------+--------------------------------------------------+

alphasparse_sctr
^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_?_sctr (
       const ALPHA_INT nz, 
       ALPHA_Number * x, 
       const ALPHA_INT *indx, 
       ALPHA_Number *y)

The ``alphasparse_?_ sctr`` execute the operation of dispersing the
elements of a compressed vector into the full storage vector:

.. math:: y[indx[i]] = x[i], i=0,1,... ,nz-1

Here ``x`` is a sparse vector in compressed format, ``y`` is a fully
stored vector. “``?``” indicates the data format, which corresponds to
the ``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. Output is the
updated ``y``. The input parameters of the function are shown below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors ``x`` and ``indx``      |
+--------------+-------------------------------------------------------+
| x            | Store as an array, length is at least ``nz``,         |
|              | contains the vector converted to full storage         |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of ``x`` that will be         |
|              | scattered,Store as an array, length is at least       |
|              | ``nz``                                                |
+--------------+-------------------------------------------------------+
| y            | Store as an array, length is at least                 |
|              | ``max(indx[i])``, Contains the updated vector element |
|              | value                                                 |
+--------------+-------------------------------------------------------+

alphasparse_doti
^^^^^^^^^^^^^^^^

.. code:: cpp

   ALPHA_Number alphasparse_?_doti ( 
       const ALPHA_INT nz, 
       const ALPHA_Number * x, 
       const ALPHA_INT *indx, 
       const ALPHA_Number *y)

The alphasparse_?_doti executes dot product operation of compressed real
number vector and full storage real number vector and return the result
value:

.. math:: res = x[0]\times y[indx[0]] + x[1]\times y[indx[1]] + ... + x[nz-1]\times y[indx[nz-1]]

``X`` is a compressed sparse vector, ``y`` is a fully stored vector.
“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface, ``s`` corresponds to float, ``d``
corresponds to double,The value of indx must be unique. Output result is
res, when ``nz``>0, Res is the result of dot product, otherwise the
value is 0. The input parameters of the function are shown in below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors x and indx              |
+--------------+-------------------------------------------------------+
| x            | Store as an array, The length is at least nz          |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of the vector x,Store as an   |
|              | array, The length is at least nz                      |
+--------------+-------------------------------------------------------+
| y            | Store as an array, The length is at least             |
|              | max(indx[i])                                          |
+--------------+-------------------------------------------------------+

alphasparse_dotci_sub
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   void alphasparse_?_dotci_sub ( 
       const ALPHA_INT nz, 
       const ALPHA_Number * x, 
       const ALPHA_INT *indx, 
       const ALPHA_Number *y,
       ALPHA_Number *dotci)

The alphasparse_?_dotci_sub performs complex numbers conjugate dot
product operation of compressed vector and real full storage vector and
return the result value:

.. math:: conjg(x[0])\times y[indx[0]] + ... + conjg(x[nz-1])\times y[indx[nz-1]]

``X`` is a sparse vector in a compressed format of complex numbers,
``y`` is a full storage of real numbers, and ``conjg(x[i])`` represents
the conjugation operation on the elements of the vector ``x``.“``?``”
indicates the data format, which corresponds to the ``ALPHA_Number`` in
the interface.There are two: first, ``c`` corresponds to float complex,
the data type of ``x`` is single-precision complex numbers, the data
type of ``y`` is float single-precision real number; second, ``z``
corresponds to double complex, the data type of ``x`` is
double-precision complex number, the data type of ``y`` is double
double-precision real number. The value of ``indx`` must be unique.
Output is ``Dotci``. The input parameters of the function are shown in
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors\ ``x`` and ``indx``     |
+--------------+-------------------------------------------------------+
| x            | Store as an array, The length is at least ``nz``      |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of the vector ``x``,Store as  |
|              | an array, The length is at least ``nz``               |
+--------------+-------------------------------------------------------+
| y            | Store as an array, The length is at least             |
|              | ``max(indx[i])``                                      |
+--------------+-------------------------------------------------------+
| dotci        | When ``nz>0``, contains the result of conjugate dot   |
|              | product of ``x`` and ``y``, otherwise value is 0      |
+--------------+-------------------------------------------------------+

alphasparse_dotui_sub
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   void alphasparse_?_dotui_sub ( 
       const ALPHA_INT nz, 
       const ALPHA_Number * x, 
       const ALPHA_INT *indx, 
       const ALPHA_Number *y,
       ALPHA_Number *dotui)

The ``alphasparse_?_dotui_sub`` performs complex numbers dot product of
compressed vector and real number full storage vector and return the
result value:

.. math:: res = x[0]\times y[indx[0]] + x[1]\times y(indx[1]) +... + x[nz-1]\times y[indx[nz-1]]

``X`` is a sparse vector in a compressed format of complex numbers,
``y`` is a full storage of real numbers vector. “``?``” indicates the
data format, which corresponds to the ``ALPHA_Number`` in the interface,
There are two ``ALPHA_Numbers``: first, ``c`` corresponds to float
complex, the data type of ``x`` is single-precision complex numbers, the
data type of ``y`` is float single-precision real number; second, ``z``
corresponds to double complex, the data type of ``x`` is
double-precision complex number, the data type of ``y`` is double
double-precision real number. The value of indx must be unique. Output
result is ``dotui``. The input parameters of the function are shown
below:

+--------------+-------------------------------------------------------+
| Input        | Description                                           |
| parameters   |                                                       |
+==============+=======================================================+
| nz           | Number of elements in vectors\ ``x`` and ``indx``     |
+--------------+-------------------------------------------------------+
| x            | Store as an array, The length is at least ``nz``      |
+--------------+-------------------------------------------------------+
| indx         | Given the element index of the vector ``x``,Store as  |
|              | an array, The length is at least ``nz``               |
+--------------+-------------------------------------------------------+
| y            | Store as an array, The length is at least             |
|              | ``max(indx[i])``                                      |
+--------------+-------------------------------------------------------+
| dotui        | When ``nz>0``, contains the result of the dot product |
|              | of ``x`` and ``y``, otherwise the value is 0          |
+--------------+-------------------------------------------------------+

DCU backend
-----------

Sparse Level1 Functions
~~~~~~~~~~~~~~~~~~~~~~~

alphasparse_dcu_axpyi
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_axpyi (
       ALPHA_INT nnz,
       const ALPHA_Number alpha,
       const ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       ALPHA_Number *y
   )

The ``alphasparse_dcu_?_axpyi`` function multiplies the sparse vector
``x`` with scalar ``alpha`` and adds the result to the dense vector
``y``, such that

.. math:: y=y+alpha\times x

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This function
stores the output result in In the vector ``y``. The input parameters of
the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of vector ``x``.        |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | scalar ``α``.                                      |
| **alpha**       |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the values of |
| **x_val**       | ``x``.                                             |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[inout]**     | array of values in dense format.                   |
| **y**           |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_doti
^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_doti (
       alphasparse_dcu_handle_t handle,
       ALPHA_INT nnz,
       const ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       const ALPHA_Number *y,
       ALPHA_Number *result,
       alphasparse_index_base_t idx_base)

Compute the dot product of a sparse vector with a dense vector.

``alphasparse_dcu_?_doti`` computes the dot product of the sparse vector
``x`` with the dense vector ``y``, such that

.. math:: result=y^Tx

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This function
stores the output result in In the vector ``y``. The input parameters of
the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of vector ``x``.        |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` values.                           |
| **x_val**       |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[in]** **y**  | array of values in dense format.                   |
+-----------------+----------------------------------------------------+
| **[out]**       | pointer to the result, can be host or device       |
| **result**      | memory                                             |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_dotci
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_dotci (
       alphasparse_dcu_handle_t handle,
       ALPHA_INT nnz,
       const ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       const ALPHA_Number *y,
       ALPHA_Number *result,
       alphasparse_index_base_t idx_base)

Compute the dot product of a complex conjugate sparse vector with a
dense vector.

``alphasparse_dcu_?_dotci`` computes the dot product of the complex
conjugate sparse vector ``x`` with the dense vector ``y``, such that

.. math:: result=x^{-H}\times y

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of vector ``x``.        |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the values of |
| **x_val**       | ``x``.                                             |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[inout]**     | array of values in dense format.                   |
| **y**           |                                                    |
+-----------------+----------------------------------------------------+
| **[out]**       | pointer to the result, can be host or device       |
| **result**      | memory                                             |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_gthr
^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_gthr(
       alphasparse_dcu_handle_t handle,
       ALPHA_INT nnz,
       const ALPHA_Number *y,
       ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       alphasparse_index_base_t idx_base)

Gather elements from a dense vector and store them into a sparse vector.

``alphasparse_dcu_?_gthr`` gathers the elements that are listed in
``x_ind`` from the dense vector ``y`` and stores them in the sparse
vector ``x``.

.. math:: x\_val[i] = y[x\_ind[i]]

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This function
stores the output result in In the vector ``y``. The input parameters of
the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of ``x``.               |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[in]** **y**  | array of values in dense format.                   |
+-----------------+----------------------------------------------------+
| **[out]**       | array of ``nnz`` elements containing the values of |
| **x_val**       | ``x``.                                             |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_gthrz
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_gthrz(
       alphasparse_dcu_handle_t handle,
       ALPHA_INT nnz,
       const ALPHA_Number *y,
       ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       alphasparse_index_base_t idx_base)

Gather and zero out elements from a dense vector and store them into a
sparse vector.

``alphasparse_dcu_?_gthrz`` gathers the elements that are listed in
``x_ind`` from the dense vector ``y`` and stores them in the sparse
vector ``x``. The gathered elements in ``y`` are replaced by zero.

.. math:: x\_val[i] = y[x\_ind[i]]

.. math:: y[x\_ind[i]]=0

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This function
stores the output result in In the vector ``y``. The input parameters of
the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of ``x``.               |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[in]** **y**  | array of values in dense format.                   |
+-----------------+----------------------------------------------------+
| **[out]**       | array of ``nnz`` elements containing the values of |
| **x_val**       | ``x``.                                             |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_roti
^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_roti(
       alphasparse_dcu_handle_t handle,
       ALPHA_INT nnz,
       ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       ALPHA_Number *y,
       const ALPHA_Number *c,
       const ALPHA_Number *s,
       alphasparse_index_base_t idx_base)

Apply Givens rotation to a dense and a sparse vector.

``alphasparse_dcu_?_roti`` applies the Givens rotation matrix GG to the
sparse vector ``x`` and the dense vector ``y``, where

.. math:: G=\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float and ``d``
corresponds to double. This function stores the output result in In the
vector ``y``. The input parameters of the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of ``x``.               |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[inout]**     | array of ``nnz`` elements containing the non-zero  |
| **x_val**       | values of ``x``.                                   |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[inout]**     | array of values in dense format.                   |
| **y**           |                                                    |
+-----------------+----------------------------------------------------+
| **[in]** **c**  | pointer to the cosine element of ``G``, can be on  |
|                 | host or device.                                    |
+-----------------+----------------------------------------------------+
| **[in]** **s**  | pointer to the sine element of ``G``, can be on    |
|                 | host or device.                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_sctr
^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_sctr(
       alphasparse_dcu_handle_t handle,
       ALPHA_INT nnz,
       const ALPHA_Number *x_val,
       const ALPHA_INT *x_ind,
       ALPHA_Number *y,
       alphasparse_index_base_t idx_base)

Scatter elements from a dense vector across a sparse vector.

``alphasparse_dcu_?_sctr`` scatters the elements that are listed in
``x_ind`` from the sparse vector ``x`` into the dense vector ``y``.
Indices of ``y`` that are not listed in ``x_ind`` remain unchanged.

.. math:: x[x\_ind[i]]=x\_val[i]

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``s`` corresponds to float, ``d``
corresponds to double, ``c`` corresponds to float complex, which is a
single-precision complex number, and ``z`` corresponds to double
complex, which is a double-precision complex number. This function
stores the output result in In the vector ``y``. The input parameters of
the function are shown below:

+-----------------+----------------------------------------------------+
| Input           | Description                                        |
| parameters      |                                                    |
+=================+====================================================+
| **[in]**        | handle to the alphasparse library context queue.   |
| **handle**      |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | number of non-zero entries of ``x``.               |
| **nnz**         |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the non-zero  |
| **x_val**       | values of ``x``.                                   |
+-----------------+----------------------------------------------------+
| **[in]**        | array of ``nnz`` elements containing the indices   |
| **x_ind**       | of the non-zero values of ``x``.                   |
+-----------------+----------------------------------------------------+
| **[inout]**     | array of values in dense format.                   |
| **y**           |                                                    |
+-----------------+----------------------------------------------------+
| **[in]**        | Indicates the addressing mode of the input         |
| **idx_base**    | array,There are the following                      |
|                 | options:\ ``ALPHA_SPARSE_INDEX_BASE_ZERO``, Based  |
|                 | on 0 addressing, the index starts with             |
|                 | 0\ ``ALPHA_SPARSE_INDEX_BASE_ONE``, Based on 1     |
|                 | addressing, the index starts with 1                |
+-----------------+----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

Sparse Level 2 Functions
~~~~~~~~~~~~~~~~~~~~~~~~

alphasparse_dcu_csrmv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrmv(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans,
       ALPHA_INT m,
       ALPHA_INT n,
       ALPHA_INT nnz,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *csr_val,
       const ALPHA_INT *csr_row_ptr,
       const ALPHA_INT *csr_col_ind,
       alphasparse_dcu_mat_info_t info,
       const ALPHA_Number *x,
       const ALPHA_Number *beta,
       ALPHA_Number *y)

Sparse matrix vector multiplication using CSR storage format.

``alphasparse_dcu_?_csrmv`` multiplies the scalar ``α`` with a sparse
``m×n`` matrix, defined in CSR storage format, and the dense vector
``x`` and adds the result to the dense vector ``y`` that is multiplied
by the scalar ``β``, such that

.. math:: y=alpha \times op(A) \times x + beta \times y

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix,     |
| **trans**         | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse CSR matrix.         |
+-------------------+--------------------------------------------------+
| **[in]** **n**    | number of columns of the sparse CSR matrix.      |
+-------------------+--------------------------------------------------+
| **[in]** **nnz**  | number of non-zero entries of the sparse CSR     |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL`` only.       |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements of the sparse CSR      |
| **csr_val**       | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``m+1`` elements that point to the      |
| **csr_row_ptr**   | start of every row of the sparse CSR matrix.     |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the column  |
| **csr_col_ind**   | indices of the sparse CSR matrix.                |
+-------------------+--------------------------------------------------+
| **[in]** **info** | NULL                                             |
+-------------------+--------------------------------------------------+
| **[in]** **x**    | array of ``n`` elements ( op(A)==A) or ``m``     |
|                   | elements ( op(A)==AT or op(A)==AH).              |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **y** | array of ``m`` elements ( op(A)==A) or n         |
|                   | elements ( op(A)==AT or op(A)==AH).              |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_coomv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrmv(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans,
       ALPHA_INT m,
       ALPHA_INT n,
       ALPHA_INT nnz,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *coo_val,
       const ALPHA_INT *coo_row_ind,
       const ALPHA_INT *coo_col_ind,
       const ALPHA_Number *x,
       const ALPHA_Number *beta,
       ALPHA_Number *y)

Sparse matrix vector multiplication using COO storage format.

``alphasparse_dcu_?_coomv`` multiplies the scalar ``α`` with a sparse
``m×n`` matrix, defined in COO storage format, and the dense vector
``x`` and adds the result to the dense vector ``y`` that is multiplied
by the scalar ``β``, such that

.. math:: y=alpha \times op(A) \times x + beta \times y

The COO matrix has to be sorted by row indices.

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix,     |
| **trans**         | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse COO matrix.         |
+-------------------+--------------------------------------------------+
| **[in]** **n**    | number of columns of the sparse COO matrix.      |
+-------------------+--------------------------------------------------+
| **[in]** **nnz**  | number of non-zero entries of the sparse COO     |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL`` only.       |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements of the sparse COO      |
| **coo_val**       | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the row     |
| **coo_row_ind**   | indices of the sparse COO matrix.                |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the column  |
| **coo_col_ind**   | indices of the sparse COO matrix.                |
+-------------------+--------------------------------------------------+
| **[in]** **x**    | array of ``n`` elements ( op(A)==A) or ``m``     |
|                   | elements ( op(A)==AT or op(A)==AH)               |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **y** | array of ``m`` elements ( op(A)==A) or n         |
|                   | elements ( op(A)==AT or op(A)==AH).              |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_ellmv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_ellmv(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans,
       ALPHA_INT m,
       ALPHA_INT n,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *ell_val,
       const ALPHA_INT *ell_col_ind,
       ALPHA_INT ell_width,
       const ALPHA_Number *x,
       const ALPHA_Number *beta,
       ALPHA_Number *y)

Sparse matrix vector multiplication using ELL storage format.

``alphasparse_dcu_?_ellmv`` multiplies the scalar ``α`` with a sparse
``m×n`` matrix, defined in ELL storage format, and the dense vector
``x`` and adds the result to the dense vector ``y`` that is multiplied
by the scalar ``β``, such that

.. math:: y=alpha \times op(A) \times x + beta \times y

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix,     |
| **trans**         | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse ELL matrix.         |
+-------------------+--------------------------------------------------+
| **[in]** **n**    | number of columns of the sparse ELL matrix.      |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL`` only.       |
+-------------------+--------------------------------------------------+
| **[in]**          | array that contains the elements of the sparse   |
| **ell_val**       | ELL matrix. Padded elements should be zero.      |
+-------------------+--------------------------------------------------+
| **[in]**          | array that contains the column indices of the    |
| **ell_col_ind**   | sparse ELL matrix. Padded column indices should  |
|                   | be -1.                                           |
+-------------------+--------------------------------------------------+
| **[in]**          | number of non-zero elements per row of the       |
| **ell_width**     | sparse ELL matrix.                               |
+-------------------+--------------------------------------------------+
| **[in]** **x**    | array of ``n`` elements ( op(A)==A) or ``m``     |
|                   | elements ( op(A)==AT or op(A)==AH)               |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **y** | array of ``m`` elements ( op(A)==A) or n         |
|                   | elements ( op(A)==AT or op(A)==AH).              |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_bsrmv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_bsrmv(
       alphasparse_dcu_handle_t handle,
       alphasparse_layout_t dir,
       alphasparse_operation_t trans,
       ALPHA_INT mb,
       ALPHA_INT nb,
       ALPHA_INT nnzb,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *bsr_val,
       const ALPHA_INT *bsr_row_ptr,
       const ALPHA_INT *bsr_col_ind,
       ALPHA_INT bsr_dim,
       const ALPHA_Number *x,
       const ALPHA_Number *beta,
       ALPHA_Number *y)

Sparse matrix vector multiplication using BSR storage format.

``alphasparse_dcu_?_bsrmv`` multiplies the scalar ``α`` with a sparse
``(mb⋅bsr_dim)×(nb⋅bsr_dim)`` matrix, defined in BSR storage format, and
the dense vector ``x`` and adds the result to the dense vector ``y``
that is multiplied by the scalar ``β``, such that

.. math:: y=alpha \times op(A) \times x + beta \times y

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix,     |
| **trans**         | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **mb**   | number of block rows of the sparse BSR matrix.   |
+-------------------+--------------------------------------------------+
| **[in]** **nb**   | number of block columns of the sparse BSR        |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]** **nnzb** | number of non-zero blocks of the sparse BSR      |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL`` only.       |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnzb`` blocks of the sparse BSR       |
| **bsr_val**       | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``mb+1`` elements that point to the     |
| **bsr_row_ptr**   | start of every block row of the sparse BSR       |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` containing the block column     |
| **bsr_col_ind**   | indices of the sparse BSR matrix.                |
+-------------------+--------------------------------------------------+
| **[in]**          | block dimension of the sparse BSR matrix.        |
| **bsr_dim**       |                                                  |
+-------------------+--------------------------------------------------+
| **[in]** **x**    | array of ``nb*bsr_dim`` elements ( op(A)==A) or  |
|                   | ``mb*bsr_dim`` elements ( op(A)==AT or           |
|                   | op(A)==AH)                                       |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **y** | array of ``mb*bsr_dim`` elements ( op(A)==A) or  |
|                   | **nb*bsr_dim** elements ( op(A)==AT or           |
|                   | op(A)==AH).                                      |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_hybmv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_hybmv(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const alphasparse_dcu_hyb_mat_t hyb,
       const ALPHA_Number *x,
       const ALPHA_Number *beta,
       ALPHA_Number *y)

Sparse matrix vector multiplication using HYB storage format.

``alphasparse_dcu_?_hybmv`` multiplies the scalar ``α`` with a sparse
``m×n`` matrix, defined in HYB storage format, and the dense vector
``x`` and adds the result to the dense vector ``y`` that is multiplied
by the scalar ``β``, such that

.. math:: y=alpha \times op(A) \times x + beta \times y

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+----------------+-----------------------------------------------------+
| Input          | Description                                         |
| parameters     |                                                     |
+================+=====================================================+
| **[in]**       | handle to the alphasparse library context queue.    |
| **handle**     |                                                     |
+----------------+-----------------------------------------------------+
| **[in]**       | For specific operations on the input matrix, there  |
| **trans**      | are the following                                   |
|                | options:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                | non-transposed,                                     |
|                | ``o                                                 |
|                | p(A) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                | transpose,                                          |
|                | ``op(A) = AT``                                      |
|                | \ \ ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                | Conjugation Transpose, ``op(A) = AH``\ Current      |
|                | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``    |
|                | only.                                               |
+----------------+-----------------------------------------------------+
| **[in]**       | scalar ``α``.                                       |
| **alpha**      |                                                     |
+----------------+-----------------------------------------------------+
| **[in]**       | This structure describes a sparse matrix with       |
| **descr**      | special structural attributes, and has three        |
|                | members:\ ``type``, ``mode``, and ``diag``. The     |
|                | ``type`` member indicates the matrix                |
|                | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,        |
|                | general                                             |
|                | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,      |
|                | diagonal                                            |
|                | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``,    |
|                | triangular                                          |
|                | mat                                                 |
|                | rix\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                | Block Triangular matrix (Only in sparse matrix      |
|                | format                                              |
|                | BSR)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``,  |
|                | Block Diagonal matrix (Only in sparse matrix BSR    |
|                | format)The ``mode`` member indicates the triangular |
|                | characteristics of the                              |
|                | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower    |
|                | triangular                                          |
|                | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper     |
|                | triangular matrix\ ``Diag`` indicates whether the   |
|                | non-zero elements of the diagonal matrix are equal  |
|                | to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all      |
|                | diagonal elements are equal to                      |
|                | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal         |
|                | elements are all equal to 1Current support          |
|                | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL`` only.          |
+----------------+-----------------------------------------------------+
| **[in]**       | matrix in HYB storage format.                       |
| **hyb**        |                                                     |
+----------------+-----------------------------------------------------+
| **[in]** **x** | array of ``n`` elements ( op(A)==A) or ``m``        |
|                | elements ( op(A)==AT or op(A)==AH)                  |
+----------------+-----------------------------------------------------+
| **[in]**       | scalar ``β``.                                       |
| **beta**       |                                                     |
+----------------+-----------------------------------------------------+
| **[inout]**    | array of ``m`` elements ( op(A)==A) or n elements ( |
| **y**          | op(A)==AT or op(A)==AH).                            |
+----------------+-----------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_csrsv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrsv_solve(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans,
       ALPHA_INT m,
       ALPHA_INT nnz,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *csr_val,
       const ALPHA_INT *csr_row_ptr,
       const ALPHA_INT *csr_col_ind,
       alphasparse_dcu_mat_info_t info,
       const ALPHA_Number *x,
       ALPHA_Number *y,
       alphasparse_dcu_solve_policy_t policy,
       void *temp_buffer)

Sparse triangular solve using CSR storage format.

``alphasparse_dcu_?_csrsv_solve`` solves a sparse triangular linear
system of a sparse ``m×m`` matrix, defined in CSR storage format, a
dense solution vector ``y`` and the right-hand side ``x`` that is
multiplied by ``α``, such that

.. math:: op(A)⋅y=α⋅x

The sparse CSR matrix has to be sorted.

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix,     |
| **trans**         | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | && ``ALPHA_SPARSE_OPERATION_TRANSPOSE``\ only.   |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse CSR matrix.         |
+-------------------+--------------------------------------------------+
| **[in]** **nnz**  | number of non-zero entries of the sparse CSR     |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR`` only.    |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements of the sparse CSR      |
| **csr_val**       | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``m+1`` elements that point to the      |
| **csr_row_ptr**   | start of every row of the sparse CSR matrix.     |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the column  |
| **csr_col_ind**   | indices of the sparse CSR matrix.                |
+-------------------+--------------------------------------------------+
| **[in]** **info** | nullptr                                          |
+-------------------+--------------------------------------------------+
| **[in]** **x**    | array of ``m`` elements, holding the right-hand  |
|                   | side.                                            |
+-------------------+--------------------------------------------------+
| **[out]** **y**   | array of ``m`` elements, holding the solution.   |
+-------------------+--------------------------------------------------+
| **[in]**          | nullptr                                          |
| **policy**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | temporary storage buffer allocated by the        |
| **temp_buffer**   | user.(nullptr)                                   |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_bsrsv
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_bsrsv_solve(
       alphasparse_dcu_handle_t handle,
       alphasparse_layout_t dir,
       alphasparse_operation_t trans,
       ALPHA_INT mb,
       ALPHA_INT nnzb,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *bsr_val,
       const ALPHA_INT *bsr_row_ptr,
       const ALPHA_INT *bsr_col_ind,
       ALPHA_INT bsr_dim,
       alphasparse_dcu_mat_info_t info,
       const ALPHA_Number *x,
       ALPHA_Number *y,
       alphasparse_dcu_solve_policy_t policy,
       void *temp_buffer)

Sparse triangular solve using BSR storage format.

``alphasparse_dcu_?_bsrsv_solve`` solves a sparse triangular linear
system of a sparse ``m×m`` matrix, defined in BSR storage format, a
dense solution vector ``y`` and the right-hand side ``x`` that is
multiplied by ``α``, such that

.. math:: op(A)⋅y=α⋅x

The sparse BSR matrix has to be sorted.

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]** **dir**  | Describe the storage mode of BSR                 |
|                   | blocks:\ ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``, row  |
|                   | major                                            |
|                   | design\ ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``,    |
|                   | column major design                              |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix,     |
| **trans**         | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | && ``ALPHA_SPARSE_OPERATION_TRANSPOSE``\ only.   |
+-------------------+--------------------------------------------------+
| **[in]** **mb**   | number of block rows of the sparse BSR matrix.   |
+-------------------+--------------------------------------------------+
| **[in]** **nnzb** | number of non-zero blocks of the sparse BSR      |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR`` only.    |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnzb`` blocks of the sparse BSR       |
| **bsr_val**       | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``mb+1`` elements that point to the     |
| **bsr_row_ptr**   | start of every block row of the sparse BSR       |
|                   | matrix.                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` containing the block column     |
| **bsr_col_ind**   | indices of the sparse BSR matrix.                |
+-------------------+--------------------------------------------------+
| **[in]**          | block dimension of the sparse BSR matrix.        |
| **bsr_dim**       |                                                  |
+-------------------+--------------------------------------------------+
| **[in]** **info** | nullptr                                          |
+-------------------+--------------------------------------------------+
| **[in]** **x**    | array of ``m`` elements, holding the right-hand  |
|                   | side.                                            |
+-------------------+--------------------------------------------------+
| **[out]** **y**   | array of ``m`` elements, holding the solution.   |
+-------------------+--------------------------------------------------+
| **[in]**          | nullptr                                          |
| **policy**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | temporary storage buffer allocated by the        |
| **temp_buffer**   | user.(nullptr)                                   |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

Sparse Level3 Functions
~~~~~~~~~~~~~~~~~~~~~~~

alphasparse_dcu_csrmm
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrmm(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans_A,
       alphasparse_operation_t trans_B,
       alphasparse_layout_t layout,
       ALPHA_INT m,
       ALPHA_INT n,
       ALPHA_INT k,
       ALPHA_INT nnz,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *csr_val,
       const ALPHA_INT *csr_row_ptr,
       const ALPHA_INT *csr_col_ind,
       const ALPHA_Number *B,
       ALPHA_INT ldb,
       const ALPHA_Number *beta,
       ALPHA_Number *matC,
       ALPHA_INT ldc)

Sparse matrix dense matrix multiplication using CSR storage format.

``alphasparse_dcu_?_csrmm`` multiplies the scalar ``α`` with a sparse
``m×k`` matrix ``A``, defined in CSR storage format, and the dense
``k×n`` matrix ``B`` and adds the result to the dense ``m×n`` matrix
``C`` that is multiplied by the scalar ``β``, such that

.. math:: C:=α⋅op(A)⋅op(B)+β⋅C

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix A,   |
| **transA**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix B,   |
| **transB**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse CSR matrix ``A``.   |
+-------------------+--------------------------------------------------+
| **[in]** **n**    | number of columns of the dense matrix ``op(B)``  |
|                   | and ``C``.                                       |
+-------------------+--------------------------------------------------+
| **[in]** **k**    | number of columns of the sparse CSR matrix       |
|                   | ``A``.                                           |
+-------------------+--------------------------------------------------+
| **[in]** **nnz**  | number of non-zero entries of the sparse CSR     |
|                   | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.      |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements of the sparse CSR      |
| **csr_val**       | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``m+1`` elements that point to the      |
| **csr_row_ptr**   | start of every row of the sparse CSR matrix      |
|                   | ``A``.                                           |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the column  |
| **csr_col_ind**   | indices of the sparse CSR matrix ``A``.          |
+-------------------+--------------------------------------------------+
| **[in]** **B**    | array of dimension ``ldb×n ( op(B)==B)`` or      |
|                   | ``ldb×k ( op(B)==BT or op(B)==BH)``.             |
+-------------------+--------------------------------------------------+
| **[in]** **ldb**  | leading dimension of ``B``, must be at least     |
|                   | ``max(1,k)`` ``( op(A)==A)`` or ``max(1,m)``     |
|                   | ``( op(A)==AT or op(A)==AH)``.                   |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **C** | array of dimension ``ldc×n``.                    |
+-------------------+--------------------------------------------------+
| **[in]** **ldc**  | leading dimension of ``C``, must be at least     |
|                   | ``max(1,m)`` ``( op(A)==Aop(A)==A)`` or          |
|                   | ``max(1,k)``                                     |
|                   | ``( op(A)==ATop(A)==AT or op(A)==AHop(A)==AH)``. |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_bsrmm
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_bsrmm(
       alphasparse_dcu_handle_t handle,
       alphasparse_layout_t dir,
       alphasparse_operation_t trans_A,
       alphasparse_operation_t trans_B,
       ALPHA_INT mb,
       ALPHA_INT n,
       ALPHA_INT kb,
       ALPHA_INT nnzb,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *bsr_val,
       const ALPHA_INT *bsr_row_ptr,
       const ALPHA_INT *bsr_col_ind,
       ALPHA_INT block_dim,
       const ALPHA_Number *matB,
       ALPHA_INT ldb,
       const ALPHA_Number *beta,
       ALPHA_Number *matC,
       ALPHA_INT ldc)

Sparse matrix dense matrix multiplication using BSR storage format.

``alphasparse_dcu_?_bsrmm`` multiplies the scalar ``α`` with a sparse
``mb×kb`` matrix ``A``, defined in BSR storage format, and the dense
``k×n`` matrix ``B`` (where ``k=block_dim×kb``) and adds the result to
the dense ``m×n`` matrix ``C`` (where ``m=block_dim×mb``) that is
multiplied by the scalar ``β``, such that

.. math:: C:=α⋅op(A)⋅op(B)+β⋅C

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]** **dir**  | Describe the storage mode of BSR                 |
|                   | blocks:\ ``ALPHA_SPARSE_LAYOUT_ROW_MAJOR``, row  |
|                   | major                                            |
|                   | design\ ``ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR``,    |
|                   | column major design                              |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix A,   |
| **transA**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix B,   |
| **transB**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **mb**   | number of block rows of the sparse BSR matrix    |
|                   | ``A``.                                           |
+-------------------+--------------------------------------------------+
| **[in]** **nn**   | number of columns of the dense matrix ``op(B)``  |
|                   | and ``C``.                                       |
+-------------------+--------------------------------------------------+
| **[in]** **kb**   | number of block columns of the sparse BSR matrix |
|                   | AA.                                              |
+-------------------+--------------------------------------------------+
| **[in]** **nnzb** | number of non-zero blocks of the sparse BSR      |
|                   | matrix AA.                                       |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.      |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnzb*block_dim*block_dim`` elements   |
| **bsr_val**       | of the sparse BSR matrix ``A``.                  |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``mb+1`` elements that point to the     |
| **bsr_row_ptr**   | start of every block row of the sparse BSR       |
|                   | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnzb`` elements containing the block  |
| **bsr_col_ind**   | column indices of the sparse BSR matrix ``A``.   |
+-------------------+--------------------------------------------------+
| **[in]**          | size of the blocks in the sparse BSR matrix.     |
| **block_dim**     |                                                  |
+-------------------+--------------------------------------------------+
| **[in]** **B**    | array of dimension ``ldb×n ( op(B)==B)`` or      |
|                   | ``ldb×k ( op(B)==BT or op(B)==BH)``.             |
+-------------------+--------------------------------------------------+
| **[in]** **ldb**  | leading dimension of ``B``, must be at least     |
|                   | ``max(1,k)`` where k=\ ``block_dim×kb``.         |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **C** | array of dimension ``ldc×n``.                    |
+-------------------+--------------------------------------------------+
| **[in]** **ldc**  | leading dimension of ``C``, must be at least     |
|                   | ``max(1,m)`` where ``m=block_dim×mb``.           |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_csrsm_solve
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrsm_solve(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans_A,
       alphasparse_operation_t trans_B,
       ALPHA_INT m,
       ALPHA_INT nrhs,
       ALPHA_INT nnz,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *csr_val,
       const ALPHA_INT *csr_row_ptr,
       const ALPHA_INT *csr_col_ind,
       ALPHA_Number *B,
       ALPHA_INT ldb,
       alphasparse_dcu_mat_info_t info,
       alphasparse_dcu_solve_policy_t policy,
       void *temp_buffer)

Sparse triangular system solve using CSR storage format.

``alphasparse_dcu_?_csrsm_solve`` solves a sparse triangular linear
system of a sparse ``m×m`` matrix, defined in CSR storage format, a
dense solution matrix ``X`` and the right-hand side matrix ``B`` that is
multiplied by ``α``, such that

.. math:: op(A)⋅op(X)=α⋅op(B)

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix A,   |
| **transA**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix B,   |
| **transB**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse CSR matrix ``A``.   |
+-------------------+--------------------------------------------------+
| **[in]** **nrhs** | number of columns of the dense matrix ``op(B)``. |
+-------------------+--------------------------------------------------+
| **[in]** **nnz**  | number of non-zero entries of the sparse CSR     |
|                   | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``\ only.   |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements of the sparse CSR      |
| **csr_val**       | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``m+1`` elements that point to the      |
| **csr_row_ptr**   | start of every row of the sparse CSR matrix      |
|                   | ``A``.                                           |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the column  |
| **csr_col_ind**   | indices of the sparse CSR matrix ``A``.          |
+-------------------+--------------------------------------------------+
| **[in]** **B**    | array of ``m`` ×× ``nrhs`` elements of the rhs   |
|                   | matrix ``B``.                                    |
+-------------------+--------------------------------------------------+
| **[in]** **ldb**  | leading dimension of rhs matrix ``B``.           |
+-------------------+--------------------------------------------------+
| **[in]** **info** | nullptr                                          |
+-------------------+--------------------------------------------------+
| **[in]**          | nullptr                                          |
| **policy**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | nullptr                                          |
| **temp_buffer**   |                                                  |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_gemmi
^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_gemmi(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans_A,
       alphasparse_operation_t trans_B,
       ALPHA_INT m,
       ALPHA_INT n,
       ALPHA_INT k,
       ALPHA_INT nnz,
       const ALPHA_Number *alpha,
       const ALPHA_Number *matA,
       ALPHA_INT lda,
       const alpha_dcu_matrix_descr_t descr,
       const ALPHA_Number *csr_val,
       const ALPHA_INT *csr_row_ptr,
       const ALPHA_INT *csr_col_ind,
       const ALPHA_Number *beta,
       ALPHA_Number *matC,
       ALPHA_INT ldc)

Dense matrix sparse matrix multiplication using CSR storage format.

``alphasparse_dcu_?_gemmi`` multiplies the scalar ``α`` with a dense
``m×k`` matrix ``A`` and the sparse ``k×n`` matrix ``B``, defined in CSR
storage format and adds the result to the dense ``m×n`` matrix ``C``
that is multiplied by the scalar ``β``, such that

.. math:: C:=α⋅op(A)⋅op(B)+β⋅C

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+-------------------+--------------------------------------------------+
| Input parameters  | Description                                      |
+===================+==================================================+
| **[in]**          | handle to the alphasparse library context queue. |
| **handle**        |                                                  |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix A,   |
| **transA**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | only.                                            |
+-------------------+--------------------------------------------------+
| **[in]**          | For specific operations on the input matrix B,   |
| **transB**        | there are the following                          |
|                   | opt                                              |
|                   | ions:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                   | non-transposed,                                  |
|                   | ``op(A                                           |
|                   | ) = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                   | transpose,                                       |
|                   | ``op(A) = AT``\ \                                |
|                   |  ``ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                   | Conjugation Transpose, ``op(A) = AH``\ Current   |
|                   | support ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` |
|                   | & ``ALPHA_SPARSE_OPERATION_TRANSPOSE`` only.     |
+-------------------+--------------------------------------------------+
| **[in]** **m**    | number of rows of the sparse CSR matrix ``A``.   |
+-------------------+--------------------------------------------------+
| **[in]** **n**    | number of columns of the dense matrix ``op(B)``  |
|                   | and ``C``.                                       |
+-------------------+--------------------------------------------------+
| **[in]** **k**    | number of columns of the sparse CSR matrix       |
|                   | ``A``.                                           |
+-------------------+--------------------------------------------------+
| **[in]** **nnz**  | number of non-zero entries of the sparse CSR     |
|                   | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | scalar ``α``.                                    |
| **alpha**         |                                                  |
+-------------------+--------------------------------------------------+
| **[in]** **A**    | array of dimension ``lda×k ( op(A)==A)`` or      |
|                   | ``lda×m ( op(A)==AT or op(A)==AH)``.             |
+-------------------+--------------------------------------------------+
| **[in]** **lda**  | leading dimension of ``A``, must be at least     |
|                   | ``m ( op(A)==A)`` or                             |
|                   | ``k ( op(A)==AT or op(A)==AH)``.                 |
+-------------------+--------------------------------------------------+
| **[in]**          | This structure describes a sparse matrix with    |
| **descr**         | special structural attributes, and has three     |
|                   | members:\ ``type``, ``mode``, and ``diag``. The  |
|                   | ``type`` member indicates the matrix             |
|                   | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,     |
|                   | general                                          |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``,   |
|                   | diagonal                                         |
|                   | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                   | triangular                                       |
|                   | matrix                                           |
|                   | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                   | Block Triangular matrix (Only in sparse matrix   |
|                   | format                                           |
|                   | BS                                               |
|                   | R)\ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                   | Block Diagonal matrix (Only in sparse matrix BSR |
|                   | format)The ``mode`` member indicates the         |
|                   | triangular characteristics of the                |
|                   | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``, lower |
|                   | triangular                                       |
|                   | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``, upper  |
|                   | triangular matrix\ ``Diag`` indicates whether    |
|                   | the non-zero elements of the diagonal matrix are |
|                   | equal to 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not |
|                   | all diagonal elements are equal to               |
|                   | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal      |
|                   | elements are all equal to 1Current support       |
|                   | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.      |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements of the sparse CSR      |
| **csr_val**       | matrix ``A``.                                    |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``m+1`` elements that point to the      |
| **csr_row_ptr**   | start of every row of the sparse CSR matrix      |
|                   | ``A``.                                           |
+-------------------+--------------------------------------------------+
| **[in]**          | array of ``nnz`` elements containing the column  |
| **csr_col_ind**   | indices of the sparse CSR matrix ``A``.          |
+-------------------+--------------------------------------------------+
| **[in]** **beta** | scalar ``β``.                                    |
+-------------------+--------------------------------------------------+
| **[inout]** **C** | array of dimension ``ldc×n`` that holds the      |
|                   | values of ``C``.                                 |
+-------------------+--------------------------------------------------+
| **[in]** **ldc**  | leading dimension of ``C``, must be at least     |
|                   | ``m``.                                           |
+-------------------+--------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

Sparse Extra Functions
~~~~~~~~~~~~~~~~~~~~~~

alphasparse_dcu_csrgeam_nnz
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_csrgeam_nnz(
       alphasparse_dcu_handle_t handle,
       ALPHA_INT m,
       ALPHA_INT n,
       const alpha_dcu_matrix_descr_t descr_A,
       ALPHA_INT nnz_A,
       const ALPHA_INT *csr_row_ptr_A,
       const ALPHA_INT *csr_col_ind_A,
       const alpha_dcu_matrix_descr_t descr_B,
       ALPHA_INT nnz_B,
       const ALPHA_INT *csr_row_ptr_B,
       const ALPHA_INT *csr_col_ind_B,
       const alpha_dcu_matrix_descr_t descr_C,
       ALPHA_INT *csr_row_ptr_C,
       ALPHA_INT *nnz_C)

Sparse matrix sparse matrix addition using CSR storage format.

``alphasparse_dcu_csrgeam_nnz`` computes the total CSR non-zero elements
and the CSR row offsets, that point to the start of every row of the
sparse CSR matrix, of the resulting matrix ``C``. It is assumed that
``csr_row_ptr_C`` has been allocated with size ``m`` + 1.

+---------------------+------------------------------------------------+
| Input parameters    | Description                                    |
+=====================+================================================+
| **[in]** **handle** | handle to the alphasparse library context      |
|                     | queue.                                         |
+---------------------+------------------------------------------------+
| **[in]** **m**      | number of rows of the sparse CSR matrix ``A``, |
|                     | ``B`` and ``C``.                               |
+---------------------+------------------------------------------------+
| **[in]** **n**      | number of columns of the sparse CSR matrix     |
|                     | ``A``, ``B`` and ``C``.                        |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``A``.This |
| **descr_A**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_A**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``A``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_A**   | start of every row of the sparse CSR matrix    |
|                     | ``A``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_A`` elements containing the     |
| **csr_col_ind_A**   | column indices of the sparse CSR matrix ``A``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``B``.This |
| **descr_B**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_B**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``B``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_B**   | start of every row of the sparse CSR matrix    |
|                     | ``B``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz`` elements containing the       |
| **csr_col_ind_B**   | column indices of the sparse CSR matrix ``B``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``C``.This |
| **descr_C**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[out]**           | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_C**   | start of every row of the sparse CSR matrix    |
|                     | ``C``.                                         |
+---------------------+------------------------------------------------+
| **[out]** **nnz_C** | pointer to the number of non-zero entries of   |
|                     | the sparse CSR matrix ``C``. ``nnz_C`` can be  |
|                     | a host or device pointer.                      |
+---------------------+------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_csrgeam
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrgeam(
       alphasparse_dcu_handle_t handle,
       ALPHA_INT m,
       ALPHA_INT n,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr_A,
       ALPHA_INT nnz_A,
       const ALPHA_Number *csr_val_A,
       const ALPHA_INT *csr_row_ptr_A,
       const ALPHA_INT *csr_col_ind_A,
       const ALPHA_Number *beta,
       const alpha_dcu_matrix_descr_t descr_B,
       ALPHA_INT nnz_B,
       const ALPHA_Number *csr_val_B,
       const ALPHA_INT *csr_row_ptr_B,
       const ALPHA_INT *csr_col_ind_B,
       const alpha_dcu_matrix_descr_t descr_C,
       ALPHA_Number *csr_val_C,
       const ALPHA_INT *csr_row_ptr_C,
       ALPHA_INT *csr_col_ind_C)

Sparse matrix sparse matrix addition using CSR storage format.

``alphasparse_dcu_?_csrgeam`` multiplies the scalar ``α`` with the
sparse ``m×n`` matrix ``A``, defined in CSR storage format, multiplies
the scalar ``β`` with the sparse ``m×n`` matrix ``B``, defined in CSR
storage format, and adds both resulting matrices to obtain the sparse
``m×n`` matrix ``C``, defined in CSR storage format, such that

.. math:: C:=α⋅A+β⋅B

It is assumed that ``csr_row_ptr_C`` has already been filled and that
``csr_val_C`` and ``csr_col_ind_C`` are allocated by the user.
``csr_row_ptr_C`` and allocation size of ``csr_col_ind_C`` and
``csr_val_C`` is defined by the number of non-zero elements of the
sparse CSR matrix ``C``. Both can be obtained by
``alphasparse_dcu_csrgeam_nnz``.

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+---------------------+------------------------------------------------+
| Input parameters    | Description                                    |
+=====================+================================================+
| **[in]** **handle** | handle to the alphasparse library context      |
|                     | queue.                                         |
+---------------------+------------------------------------------------+
| **[in]** **m**      | number of rows of the sparse CSR matrix ``A``, |
|                     | ``B`` and ``C``.                               |
+---------------------+------------------------------------------------+
| **[in]** **n**      | number of columns of the sparse CSR matrix     |
|                     | ``A``, ``B`` and ``C``.                        |
+---------------------+------------------------------------------------+
| **[in]** **alpha**  | scalar ``α``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``A``.This |
| **descr_A**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_A**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``A``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_A`` elements of the sparse CSR  |
| **csr_val_A**       | matrix ``A``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_A**   | start of every row of the sparse CSR matrix    |
|                     | ``A``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_A`` elements containing the     |
| **csr_col_ind_A**   | column indices of the sparse CSR matrix ``A``. |
+---------------------+------------------------------------------------+
| **[in]** **beta**   | scalar ``β``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``B``.This |
| **descr_B**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_B**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``B``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_B`` elements of the sparse CSR  |
| **csr_val_B**       | matrix ``B``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_B**   | start of every row of the sparse CSR matrix    |
|                     | ``B``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz`` elements containing the       |
| **csr_col_ind_B**   | column indices of the sparse CSR matrix ``B``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``C``.This |
| **descr_C**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[out]**           | array of elements of the sparse CSR matrix     |
| **csr_val_C**       | ``C``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_C**   | start of every row of the sparse CSR matrix    |
|                     | ``C``.                                         |
+---------------------+------------------------------------------------+
| **[out]**           | array of elements containing the column        |
| **csr_col_ind_C**   | indices of the sparse CSR matrix ``C``.        |
+---------------------+------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

Both scalars ``α`` and ``beta`` have to be valid.

alphasparse_dcu_csrgemm_nnz
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_csrgemm_nnz(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans_A,
       alphasparse_operation_t trans_B,
       ALPHA_INT m,
       ALPHA_INT n,
       ALPHA_INT k,
       const alpha_dcu_matrix_descr_t descr_A,
       ALPHA_INT nnz_A,
       const ALPHA_INT *csr_row_ptr_A,
       const ALPHA_INT *csr_col_ind_A,
       const alpha_dcu_matrix_descr_t descr_B,
       ALPHA_INT nnz_B,
       const ALPHA_INT *csr_row_ptr_B,
       const ALPHA_INT *csr_col_ind_B,
       const alpha_dcu_matrix_descr_t descr_D,
       ALPHA_INT nnz_D,
       const ALPHA_INT *csr_row_ptr_D,
       const ALPHA_INT *csr_col_ind_D,
       const alpha_dcu_matrix_descr_t descr_C,
       ALPHA_INT *csr_row_ptr_C,
       ALPHA_INT *nnz_C,
       const alphasparse_dcu_mat_info_t info_C,
       void *temp_buffer)

Sparse matrix sparse matrix multiplication using CSR storage format.

``alphasparse_dcu_csrgemm_nnz`` computes the total CSR non-zero elements
and the CSR row offsets, that point to the start of every row of the
sparse CSR matrix, of the resulting multiplied matrix C. It is assumed
that ``csr_row_ptr_C`` has been allocated with size ``m`` + 1.

+---------------------+------------------------------------------------+
| Input parameters    | Description                                    |
+=====================+================================================+
| **[in]** **handle** | handle to the alphasparse library context      |
|                     | queue.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | matrix ``A`` operation type.For specific       |
| **trans_A**         | operations on the input matrix A, there are    |
|                     | the following                                  |
|                     | optio                                          |
|                     | ns:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                     | non-transposed,                                |
|                     | ``op(A)                                        |
|                     | = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                     | transpose,                                     |
|                     | ``op(A) = AT``\ \ `                            |
|                     | `ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                     | Conjugation Transpose, ``op(A) = AH``\ Current |
|                     | support                                        |
|                     | ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` only. |
+---------------------+------------------------------------------------+
| **[in]**            | matrix ``B`` operation type.For specific       |
| **trans_B**         | operations on the input matrix A, there are    |
|                     | the following                                  |
|                     | optio                                          |
|                     | ns:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                     | non-transposed,                                |
|                     | ``op(A)                                        |
|                     | = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                     | transpose,                                     |
|                     | ``op(A) = AT``\ \ `                            |
|                     | `ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                     | Conjugation Transpose, ``op(A) = AH``\ Current |
|                     | support                                        |
|                     | ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` only. |
+---------------------+------------------------------------------------+
| **[in]** **m**      | number of rows of the sparse CSR matrix ``A``  |
|                     | and ``C``.                                     |
+---------------------+------------------------------------------------+
| **[in]** **n**      | number of columns of the sparse CSR matrix     |
|                     | ``B`` and ``C``.                               |
+---------------------+------------------------------------------------+
| **[in]** **k**      | number of columns of the sparse CSR matrix     |
|                     | ``op(A)`` and number of rows of the sparse CSR |
|                     | matrix ``op(B)``.                              |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``A``.This |
| **descr_A**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_A**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``A``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_A**   | start of every row of the sparse CSR matrix    |
|                     | ``A``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_A`` elements containing the     |
| **csr_col_ind_A**   | column indices of the sparse CSR matrix ``A``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``B``.This |
| **descr_B**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_B**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``B``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_B**   | start of every row of the sparse CSR matrix    |
|                     | ``B``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz`` elements containing the       |
| **csr_col_ind_B**   | column indices of the sparse CSR matrix ``B``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``D``.This |
| **descr_D**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_D**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``D``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_D**   | start of every row of the sparse CSR matrix    |
|                     | ``D``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz`` elements containing the       |
| **csr_col_ind_D**   | column indices of the sparse CSR matrix ``D``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``C``.This |
| **descr_C**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[out]**           | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_C**   | start of every row of the sparse CSR matrix    |
|                     | ``C``.                                         |
+---------------------+------------------------------------------------+
| **[out]** **nnz_C** | pointer to the number of non-zero entries of   |
|                     | the sparse CSR matrix ``C``. ``nnz_C`` can be  |
|                     | a host or device pointer.                      |
+---------------------+------------------------------------------------+
| **[in]** **info_C** | nullptr                                        |
+---------------------+------------------------------------------------+
| **[in]**            | nullptr                                        |
| **temp_buffer**     |                                                |
+---------------------+------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

alphasparse_dcu_csrgemm
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

   alphasparse_status_t alphasparse_dcu_?_csrgemm(
       alphasparse_dcu_handle_t handle,
       alphasparse_operation_t trans_A,
       alphasparse_operation_t trans_B,
       ALPHA_INT m,
       ALPHA_INT n,
       ALPHA_INT k,
       const ALPHA_Number *alpha,
       const alpha_dcu_matrix_descr_t descr_A,
       ALPHA_INT nnz_A,
       const ALPHA_Number *csr_val_A,
       const ALPHA_INT *csr_row_ptr_A,
       const ALPHA_INT *csr_col_ind_A,
       const alpha_dcu_matrix_descr_t descr_B,
       ALPHA_INT nnz_B,
       const ALPHA_Number *csr_val_B,
       const ALPHA_INT *csr_row_ptr_B,
       const ALPHA_INT *csr_col_ind_B,
       const ALPHA_Number *beta,
       const alpha_dcu_matrix_descr_t descr_D,
       ALPHA_INT nnz_D,
       const ALPHA_Number *csr_val_D,
       const ALPHA_INT *csr_row_ptr_D,
       const ALPHA_INT *csr_col_ind_D,
       const alpha_dcu_matrix_descr_t descr_C,
       ALPHA_Number *csr_val_C,
       const ALPHA_INT *csr_row_ptr_C,
       ALPHA_INT *csr_col_ind_C,
       const alphasparse_dcu_mat_info_t info_C,
       void *temp_buffer)

Sparse matrix sparse matrix multiplication using CSR storage format.

``alphasparse_dcu_?_csrgemm`` multiplies the scalar ``α`` with the
sparse ``m×k`` matrix ``A``, defined in CSR storage format, and the
sparse ``k×n`` matrix ``B``, defined in CSR storage format, and adds the
result to the sparse ``m×n`` matrix ``D`` that is multiplied by ``β``.
The final result is stored in the sparse ``m×n`` matrix ``C``, defined
in CSR storage format, such that

.. math:: C:=α⋅op(A)⋅op(B)+β⋅D

It is assumed that ``csr_row_ptr_C`` has already been filled and that
``csr_val_C`` and ``csr_col_ind_C`` are allocated by the user.
``csr_row_ptr_C`` and allocation size of ``csr_col_ind_C`` and
``csr_val_C`` is defined by the number of non-zero elements of the
sparse CSR matrix C. Both can be obtained by
``alphasparse_dcu_csrgemm_nnz``.

“``?``” indicates the data format, which corresponds to the
``ALPHA_Number`` in the interface. ``c`` corresponds to float complex,
which is a single-precision complex number and ``z`` corresponds to
double complex, which is a double-precision complex number. This
function stores the output result in In the vector ``y``. The input
parameters of the function are shown below:

+---------------------+------------------------------------------------+
| Input parameters    | Description                                    |
+=====================+================================================+
| **[in]** **handle** | handle to the alphasparse library context      |
|                     | queue.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | matrix ``A`` operation type.For specific       |
| **trans_A**         | operations on the input matrix A, there are    |
|                     | the following                                  |
|                     | optio                                          |
|                     | ns:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                     | non-transposed,                                |
|                     | ``op(A)                                        |
|                     | = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                     | transpose,                                     |
|                     | ``op(A) = AT``\ \ `                            |
|                     | `ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                     | Conjugation Transpose, ``op(A) = AH``\ Current |
|                     | support                                        |
|                     | ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` only. |
+---------------------+------------------------------------------------+
| **[in]**            | matrix ``B`` operation type.For specific       |
| **trans_B**         | operations on the input matrix A, there are    |
|                     | the following                                  |
|                     | optio                                          |
|                     | ns:\ ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE``, |
|                     | non-transposed,                                |
|                     | ``op(A)                                        |
|                     | = A``\ \ ``ALPHA_SPARSE_OPERATION_TRANSPOSE``, |
|                     | transpose,                                     |
|                     | ``op(A) = AT``\ \ `                            |
|                     | `ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE``, |
|                     | Conjugation Transpose, ``op(A) = AH``\ Current |
|                     | support                                        |
|                     | ``ALPHA_SPARSE_OPERATION_NON_TRANSPOSE`` only. |
+---------------------+------------------------------------------------+
| **[in]** **m**      | number of rows of the sparse CSR matrix ``A``  |
|                     | and ``C``.                                     |
+---------------------+------------------------------------------------+
| **[in]** **n**      | number of columns of the sparse CSR matrix     |
|                     | ``B`` and ``C``.                               |
+---------------------+------------------------------------------------+
| **[in]** **k**      | number of columns of the sparse CSR matrix     |
|                     | ``op(A)`` and number of rows of the sparse CSR |
|                     | matrix ``op(B)``.                              |
+---------------------+------------------------------------------------+
| **[in]** **alpha**  | scalar ``α``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``A``.This |
| **descr_A**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_A**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``A``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_A`` elements of the sparse CSR  |
| **csr_val_A**       | matrix ``A``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_A**   | start of every row of the sparse CSR matrix    |
|                     | ``A``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_A`` elements containing the     |
| **csr_col_ind_A**   | column indices of the sparse CSR matrix ``A``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``B``.This |
| **descr_B**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_B**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``B``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_B`` elements of the sparse CSR  |
| **csr_val_B**       | matrix ``B``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_B**   | start of every row of the sparse CSR matrix    |
|                     | ``B``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz`` elements containing the       |
| **csr_col_ind_B**   | column indices of the sparse CSR matrix ``B``. |
+---------------------+------------------------------------------------+
| **[in]** **beta**   | scalar ``β``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``D``.This |
| **descr_D**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[in]** **nnz_D**  | number of non-zero entries of the sparse CSR   |
|                     | matrix ``D``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz_D`` elements of the sparse CSR  |
| **csr_val_D**       | matrix ``D``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_D**   | start of every row of the sparse CSR matrix    |
|                     | ``D``.                                         |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``nnz`` elements containing the       |
| **csr_col_ind_D**   | column indices of the sparse CSR matrix ``D``. |
+---------------------+------------------------------------------------+
| **[in]**            | descriptor of the sparse CSR matrix ``C``.This |
| **descr_C**         | structure describes a sparse matrix with       |
|                     | special structural attributes, and has three   |
|                     | members:\ ``type``, ``mode``, and ``diag``.    |
|                     | The ``type`` member indicates the matrix       |
|                     | type:\ ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``,   |
|                     | general                                        |
|                     | matrix\ ``ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL``, |
|                     | diagonal                                       |
|                     | ma                                             |
|                     | trix\ ``ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR``, |
|                     | triangular                                     |
|                     | matrix\                                        |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR``, |
|                     | Block Triangular matrix (Only in sparse matrix |
|                     | format                                         |
|                     | BSR)                                           |
|                     | \ ``ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL``, |
|                     | Block Diagonal matrix (Only in sparse matrix   |
|                     | BSR format)The ``mode`` member indicates the   |
|                     | triangular characteristics of the              |
|                     | matrix:\ ``ALPHA_SPARSE_FILL_MODE_LOWER``,     |
|                     | lower triangular                               |
|                     | matrix\ ``ALPHA_SPARSE_FILL_MODE_UPPER``,      |
|                     | upper triangular matrix\ ``Diag`` indicates    |
|                     | whether the non-zero elements of the diagonal  |
|                     | matrix are equal to                            |
|                     | 1:\ ``ALPHA_SPARSE_DIAG_NON_UNIT``, not all    |
|                     | diagonal elements are equal to                 |
|                     | 1\ ``ALPHA_SPARSE_DIAG_UNIT``, the diagonal    |
|                     | elements are all equal to 1Current support     |
|                     | ``ALPHA_SPARSE_MATRIX_TYPE_GENERAL``\ only.    |
+---------------------+------------------------------------------------+
| **[out]**           | array of ``nnz_C`` elements of the sparse CSR  |
| **csr_val_C**       | matrix ``C``.                                  |
+---------------------+------------------------------------------------+
| **[in]**            | array of ``m+1`` elements that point to the    |
| **csr_row_ptr_C**   | start of every row of the sparse CSR matrix    |
|                     | ``C``.                                         |
+---------------------+------------------------------------------------+
| **[out]**           | array of ``nnz_C`` elements containing the     |
| **csr_col_ind_C**   | column indices of the sparse CSR matrix ``C``. |
+---------------------+------------------------------------------------+
| **[in]** **info_C** | nullptr                                        |
+---------------------+------------------------------------------------+
| **[in]**            | nullptr                                        |
| **temp_buffer**     |                                                |
+---------------------+------------------------------------------------+

This function is non blocking and executed asynchronously with respect
to the host. It may return before the actual computation has finished.

If ``α==0``, then ``C=β⋅D`` will be computed.

If ``β==0``, then ``C=α⋅op(A)⋅op(B)`` will be computed.

``α==beta==0`` is invalid.
