import unittest
from unittest.mock import patch, mock_open, MagicMock
from similarityCalculation import similarityCalculation

class TestSimilarityCalculation(unittest.TestCase):
    
    def setUp(self):
        self.data_base_files_path = "unit_test/mock_data_base"
        self.suspected_files_path = "unit_test/mock_suspected_files"
        self.suspected_tp = "unit_test/mock_suspected_files/tp"
        self.suspected_tn = "unit_test/mock_suspected_files/tn"
        self.percentaje_simil = 0.7
        self.similarity_calculator = similarityCalculation(self.data_base_files_path, self.percentaje_simil)
                
    def test_dataBaseProcessing(self):
        expected_files_and_content = {'file2.txt': 'another test file', 
                                      'file1.txt': 'test file'}
        
        with patch('similarityCalculation.similarityCalculation._uploadDatabase', return_value=expected_files_and_content):
            files_and_content_processed = self.similarity_calculator.dataBaseProcessing()
        
        self.assertDictEqual(files_and_content_processed, expected_files_and_content)
    
    @patch('similarityCalculation.similarityCalculation._uploadDatabase')
    def test_uploadDatabase_exception(self, mock_upload_database):
        mock_upload_database.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.similarity_calculator.dataBaseProcessing()
            
    def test_preprocess_text(self):
        text = "This is a test sentence."
        expected_preprocess_text = "test sentence"
        
        preprocess_text = self.similarity_calculator._preprocess_text(text)
        
        self.assertEqual(preprocess_text, expected_preprocess_text)
    
    def test_plagiarismDetection_positive(self):
        input_file_path = f'{self.suspected_files_path}/TP-01.txt'
        
        result = self.similarity_calculator.plagiarismDetection(input_file_path)

        self.assertTrue(result[0])
        self.assertIs(type(result), tuple)
        self.assertIs(type(result[2]), float)
        self.assertEqual(result[1], 'org-03.txt')
        self.assertGreaterEqual(result[2], self.percentaje_simil)
    
    def test_plagiarismDetection_negative(self):
        input_file_path = f'{self.suspected_files_path}/TN-01.txt'
        
        result = self.similarity_calculator.plagiarismDetection(input_file_path)
        
        self.assertFalse(result)
        self.assertIs(type(result), bool)

    def test_read_file(self):
        mock_file_path = f'{self.data_base_files_path}/org-01.txt'
        mock_content = "This is a mock file content."
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            content = self.similarity_calculator._read_file(mock_file_path)
        
        self.assertEqual(content, mock_content)

    def test_similarityComparison(self):
        preprocessed_input_text = "second element one preprocessed"
        expected_similar_file = 'org-02.txt'
        
        mock_files_and_content = {
            'org-01.txt': 'Tirst value to evaluate',
            'org-02.txt': 'Second element. This one has to be preprocessed',
            'org-02.txt': 'Third value. You\'ve missed the good one!'
        }
        
        result_file, result_score = self.similarity_calculator.similarityComparison(preprocessed_input_text, mock_files_and_content)
        
        self.assertEqual(result_file, expected_similar_file)
        self.assertIs(type(result_score), float)


    def test_evaluate_directory_all_tp(self):
        auc = self.similarity_calculator.evaluate_directory(self.suspected_tp)
        
        # We expect the AUC to be 1 because all files are correctly classified.
        self.assertEqual(auc, 1.0)

    def test_evaluate_directory_all_tn(self):
        auc = self.similarity_calculator.evaluate_directory(self.suspected_tn)
        
        # We expect the AUC to be 0.5 because all files are correctly classified.
        self.assertEqual(auc, 0.5)

    def test_evaluate_directory_mixed(self):
        auc = self.similarity_calculator.evaluate_directory(self.suspected_files_path)
        
        # We expect the AUC to be 1 because all files are correctly classified.
        self.assertEqual(auc, 1)
    
    
if __name__ == '__main__':
    unittest.main()
