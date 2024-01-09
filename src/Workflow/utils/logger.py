import logging

def get_default_logger(name: str, level=logging.DEBUG):
  logger = logging.getLogger(name)
  logger.setLevel(level)
  # 创建一个处理器（例如StreamHandler，用于控制台输出）
  console_handler = logging.StreamHandler()
  # 设置处理器的级别（只有达到或高于此级别的消息才会被发送到处理器）
  console_handler.setLevel(level)
  # 格式化处理
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  console_handler.setFormatter(formatter)
  # 将处理器添加到Logger中
  logger.addHandler(console_handler)
  return logger