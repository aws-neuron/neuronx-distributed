# Standard Library
import unittest
from unittest.mock import patch, MagicMock
import logging

from neuronx_distributed.utils.logger import get_log_level, get_logger, _rank0_only


class TestLogger(unittest.TestCase):
    def test_log_level(self):
        levels = {
            "trace": logging.DEBUG,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "fatal": logging.FATAL,
            "off": logging.FATAL + 1,
            "unsupported": ValueError,
        }
        for value, level in levels.items():
            with patch.dict("os.environ", {"NXD_LOG_LEVEL": value}):
                get_log_level.cache_clear()
                if not isinstance(level, int):
                    with self.assertRaises(level):
                        get_log_level()
                else:
                    self.assertEqual(level, get_log_level())

    def test_logger(self):
        for level in ["info", "off"]:
            with patch.dict("os.environ", {"NXD_LOG_LEVEL": level}), patch("logging.getLogger") as mock_logger:
                # Arrange
                get_log_level.cache_clear()
                mock_logger.return_value.initialized = False
                lvls = ["debug", "info", "warning", "error", "exception", "fatal", "critical"]
                original = {lvl: getattr(mock_logger.return_value, lvl) for lvl in lvls}
                # Act
                logger = get_logger("some_name")
                # Assert
                self.assertEqual(logger, mock_logger.return_value)
                if level == "off":
                    logger.setLevel.assert_not_called()
                    self.assertTrue(logger.disabled)
                else:
                    logger.setLevel.assert_called_once_with(get_log_level())
                    for lvl, method in original.items():
                        self.assertNotEqual(getattr(logger, lvl), method)
                self.assertFalse(logger.propagate)
                self.assertTrue(logger.initialized)

    @patch.dict("os.environ", {"NXD_LOG_LEVEL": "info"})
    @patch("logging.getLogger")
    def test_logger_skips_initialized(self, mock_logger):
        # Arrange
        mock_logger.return_value.initialized = True
        # Act
        logger = get_logger()
        # Assert
        self.assertEqual(logger, mock_logger.return_value)
        logger.addHandler.assert_not_called()

    @patch.dict("os.environ", {"NXD_LOG_LEVEL": "info"})
    @patch("logging.getLogger")
    def test_logger_not_rank0_only(self, mock_logger):
        # Arrange
        lvls = ["debug", "info", "warning", "error", "exception", "fatal", "critical"]
        original = {lvl: getattr(mock_logger.return_value, lvl) for lvl in lvls}
        # Act
        logger = get_logger(rank0_only=False)
        # Assert
        self.assertEqual(logger, mock_logger.return_value)
        for lvl, method in original.items():
            self.assertEqual(getattr(logger, lvl), method)

    @patch("torch.distributed")
    def test_rank0_works(self, mock_dist):
        # Arrange
        fn = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        # Act
        wrapped = _rank0_only(fn)
        # Assert
        self.assertEqual(wrapped(), fn.return_value)
        fn.assert_called_once_with(stacklevel=2)

    @patch.dict("os.environ", {"RANK": "1"})
    @patch("torch.distributed")
    def test_rank0_checks_torch_dist(self, mock_dist):
        # Arrange
        fn = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 42
        # Act
        wrapped = _rank0_only(fn)
        # Assert
        self.assertNotEqual(wrapped(), fn.return_value)
        fn.assert_not_called()

    @patch.dict("os.environ", {"RANK": "42"})
    @patch("torch.distributed")
    def test_rank0_checks_environ(self, mock_dist):
        # Arrange
        fn = MagicMock()
        mock_dist.is_initialized.return_value = False
        # Act
        wrapped = _rank0_only(fn)
        # Assert
        self.assertNotEqual(wrapped(), fn.return_value)
        fn.assert_not_called()

    @patch("torch.distributed")
    def test_rank0_only_assumes_rank0(self, mock_dist):
        # Arrange
        fn = MagicMock()
        mock_dist.is_initialized.return_value = False
        # Act
        wrapped = _rank0_only(fn)
        # Assert
        self.assertEqual(wrapped(), fn.return_value)
        fn.assert_called_once_with(stacklevel=2)
