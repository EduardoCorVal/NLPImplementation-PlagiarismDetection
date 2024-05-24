import unittest
import tempfile
import os
from unittest.mock import patch
from similarityCalculation import similarityCalculation

class TestSimilarityCalculation(unittest.TestCase):
    
    def setUp(self):
        self.files_path = "test_files/data_base"
        self.percentaje_simil = 0.7
        
        self.similarity_calculator = similarityCalculation(self.files_path, self.percentaje_simil)
        
    def test_dataBaseProcessing(self):
        expected_files_and_content = {'file2.txt': 'This is another test file .', 
                                      'file1.txt': 'This is a test file .'}
        
        files_and_content_processed = self.similarity_calculator.dataBaseProcessing()
        
        self.assertDictEqual(files_and_content_processed, expected_files_and_content)
    
    def test_lemmatize_text(self):
        text = "This is a test sentence."
        expected_lemmatized_text = "This is a test sentence ."
        
        lemmatized_text = self.similarity_calculator._lemmatize_text(text)
        
        self.assertEqual(lemmatized_text, expected_lemmatized_text)
    
    def test_plagiarismDetection_possitive(self):
        input_file_path = 'test_files/data_base/file1.txt'
        
        result = self.similarity_calculator.plagiarismDetection(input_file_path)

        self.assertTrue(result)
        self.assertIs(type(result), tuple)
        self.assertIs(type(result[2]), float)
        self.assertEqual(result[1], 'file1.txt')
        
    def test_plagiarismDetection_negative(self):
        input_file_path = 'test_files/external_files/file3.txt'
        
        result = self.similarity_calculator.plagiarismDetection(input_file_path)
        
        self.assertFalse(result)
        self.assertIs(type(result), bool)
    
    @patch('similarityCalculation.similarityCalculation._uploadDatabase')
    def test_uploadDatabase_exception(self, mock_upload_database):
        mock_upload_database.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.similarity_calculator.dataBaseProcessing()

if __name__ == '__main__':
    unittest.main()
