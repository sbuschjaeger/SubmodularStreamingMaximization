
.. _program_listing_file_include_DataTypeHandling.h:

Program Listing for File DataTypeHandling.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_DataTypeHandling.h>` (``include/DataTypeHandling.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef DATATYPEHANDLING_H_
   #define DATATYPEHANDLING_H_
   
   #include <vector>
   
   // Float data-typed used in the entire project. If you find a hardcoded "float" / "double" its probably a good idea to replace it with data_t
   typedef double data_t;
   
   // Index data-typed used in the entire project. If you find a hardcoded "size_t" / "unsigned int" etc. its probably a good idea to replace it with idx_t
   typedef long idx_t;
   
   #endif
