import threading
import time
from typing import Dict, Any, Optional, Tuple
import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from imageset import ImageSet

class CallbackFlag(Enum):
    NONE = 0
    CONTINUE = 1
    FINISH = 2

# ヘルパー関数（スレッド間で共有）
def _load_file_thread(shared_resources, file_path, exif_data, param, imgset):
    """
    ファイル読み込みスレッド
    
    Args:
        shared_resources: 共有リソース辞書
        file_path: ファイル名
        exif_data: EXIFデータ
        param: 追加パラメータ
    """
    try:
        # 共有リソースを解凍
        cache = shared_resources['cache']
        preload_registry = shared_resources['preload_registry']
        active_processes = shared_resources['active_processes']
        callback_flags = shared_resources['callback_flags']
        
        logging.info(f"Loading thread started for {file_path}")
        if imgset is None:
            imgset = ImageSet()
        result = imgset.load(file_path, exif_data, param)
        # 失敗？
        if result == False:
            pass

        # 成功？
        elif result == True:
            # キャッシュに登録
            cache[file_path] = (imgset, exif_data, param)

        else:
            # 中間結果をキャッシュに登録
            cache[file_path] = (imgset, exif_data, param)
            callback_flags[file_path] = CallbackFlag.CONTINUE   # 中間結果を表示

            # 続きの実行
            imgset.load(file_path, exif_data, param, result)

            # キャッシュに登録
            cache[file_path] = (imgset, exif_data, param)
        
        # 先行読み込み登録から削除
        if file_path in preload_registry:
            del preload_registry[file_path]
        
        # 完了フラグを設定
        callback_flags[file_path] = CallbackFlag.FINISH
                
    except Exception as e:
        logging.error(f"Loading file {file_path}: {e}")
        
    finally:        
        # 処理キューフラグを設定
        shared_resources['process_queue_flag'] = True

class FileCacheSystem:
    def __init__(self, max_cache_size: int = 10, max_concurrent_loads: int = 4):
        # 共有リソースを初期化
        self.shared_resources = {
            'cache': {},
            'preload_registry': {},
            'active_processes': {},
            'callback_flags': {},
            'process_queue_flag': False
        }
        
        # 各共有リソースへの参照を設定
        self.cache = self.shared_resources['cache']
        self.preload_registry = self.shared_resources['preload_registry']
        self.active_processes = self.shared_resources['active_processes']
        self.file_callbacks = {}  # コールバックはpickle化できないので共有しない
        
        # その他の設定
        self.max_cache_size = max_cache_size
        self.max_concurrent_loads = max_concurrent_loads
        
        # 監視スレッドの開始
        self.monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self.monitor_thread.start()

        # ThreadPool
        self.p = ThreadPoolExecutor(max_workers=max_concurrent_loads)
     
    def _monitor_processes(self):
        """スレッドと完了フラグを監視するスレッド"""
        while True:
            try:
                # コールバックの実行
                callback_flags = self.shared_resources['callback_flags']
                for file_path in list(callback_flags.keys()):
                    if callback_flags[file_path] != CallbackFlag.NONE:
                        if file_path in self.cache:
                            # キャッシュからデータを取得
                            imgset, exif_data, param = self.cache[file_path]

                            elapsed_time = time.time() - self.active_processes[file_path]
                            if callback_flags[file_path] == CallbackFlag.CONTINUE:
                                logging.info(f"Continue loading {file_path} 経過時間 {elapsed_time:.3f} 秒.")
                            else:
                                logging.info(f"Finish loading {file_path} 経過時間 {elapsed_time:.3f} 秒.")

                            if file_path in self.file_callbacks:
                                # コールバックを実行
                                callback = self.file_callbacks[file_path]
                                callback(file_path, imgset, exif_data, param, callback_flags[file_path])
                        
                        # 進行中のスレッドから削除
                        if callback_flags[file_path] == CallbackFlag.FINISH and file_path in self.active_processes:
                            del self.active_processes[file_path]

                        # フラグをクリア
                        del callback_flags[file_path]
                
                # キュー処理
                self.process_preload_queue(self.max_concurrent_loads)

            except Exception as e:
                logging.error(f"Error in monitor thread: {e}")
            
            time.sleep(0.01)
     
    def get_file(self, file_path: str, callback=None) -> Tuple[Dict[str, Any], Optional[ImageSet]]:
        """
        ファイルを取得する関数
        
        Args:
            file_path: ファイル名
            callback: ファイルが読み込まれた際に呼び出すコールバック関数
            
        Returns:
            Tuple[Dict[str, Any], Optional[Imgset]]: (exif_data, imgset)のタプル
            
        Raises:
            FileNotFoundError: キャッシュにも先行読み込み登録もされていない場合
        """
        # コールバックが指定された場合、登録
        if callback:
            self.file_callbacks.clear()     # 登録できるのは一つだけ
            self.file_callbacks[file_path] = callback
        
        # キャッシュにある場合
        if file_path in self.cache:
            logging.info(f"Cache hit: {file_path}")
            imgset, exif_data, param = self.cache[file_path]
            
            # コールバックが指定されていればすぐに呼び出す
            if callback:
                callback(file_path, imgset, exif_data, param)
                
            return exif_data, imgset
                
        # 先行読み込み登録がある場合
        elif file_path in self.preload_registry:
            logging.info(f"Preload registry hit: {file_path}")
            exif_data, param, imgset = self.preload_registry[file_path]
            
            # まだ読み込みスレッドが開始されていなければ開始
            if file_path not in self.active_processes:
                self._start_loading_thread(file_path, exif_data, param, imgset)
                
            return exif_data, imgset  # imgsetはまだ利用不可
        
        # どちらにもない場合
        else:
            raise FileNotFoundError(f"File {file_path} is not in cache or preload registry")
    
    def register_for_preload(self, file_path: str, exif_data: Dict[str, Any], param: Dict[str, Any] = None, 
                        high_priority: bool = False):
        """
        先行読み込み登録関数
        
        Args:
            file_path: ファイル名
            exif_data: EXIFデータ
            param: 追加パラメータ
            high_priority: 優先度が高いかどうか
        """
        if param is None:
            param = {}
            
        # すでにキャッシュにある場合は何もしない
        if file_path in self.cache:
            return
        
        # すでに先行読み込み登録されている場合は何もしない
        if file_path in self.preload_registry:
            return
        
        # 高速化のためここで作っとく
        imgset = ImageSet()
        imgset.file_path = file_path
        imgset.param = param
        
        # 先行読み込み登録
        self.preload_registry[file_path] = (exif_data, param, imgset)
        logging.info(f"Registered {file_path} for preload")
        
        # 優先度が高い場合はすぐに読み込みを開始
        if high_priority:
            self._start_loading_thread(file_path, exif_data, param, imgset)
        else:
            # 優先度が低い場合は自動的にキューを処理
            self.process_preload_queue(max_concurrent_loads=self.max_concurrent_loads)

    def _start_loading_thread(self, file_path: str, exif_data: Dict[str, Any], param: Dict[str, Any] = None, imgset: ImageSet = None):
        """読み込みスレッドを開始する内部関数"""
        if param is None:
            param = {}
        
        if file_path in self.active_processes:
            return
        
        self.active_processes[file_path] = time.time()  # プロセスIDのみ保存

        # プロセスを起動（self自体は渡さない）
        if False:
            """
            process = multiprocessing.Process(
                target=_load_file_process,
                args=(self.shared_resources, file_path, exif_data, param)
            )
            process.start()
            """
            thread = threading.Thread(target=_load_file_thread, args=[self.shared_resources, file_path, exif_data, param, imgset], daemon=True)
            thread.start()
        else:
            future = self.p.submit(_load_file_thread, self.shared_resources, file_path, exif_data, param, imgset)

            #process = self.p.apply_async(_load_file_process, args=(self.shared_resources, file_path, exif_data, param))

        logging.info(f"Started loading process for {file_path}")
            
    def clear_cache(self, keep_files=None):
        """
        キャッシュをクリアする関数
        
        Args:
            keep_files: キャッシュに残すファイル名のリスト
        """
        if keep_files is None:
            keep_files = []
        
        # キャッシュからkeep_files以外の全てのアイテムを削除
        for file_path in list(self.cache.keys()):
            if file_path not in keep_files:
                delete_cache(self.cache, file_path)
    
    def get_cache_status(self):
        """
        キャッシュの状態を取得する関数
        
        Returns:
            Dict: キャッシュの状態
        """
        return {
            "cache_size": len(self.cache),
            "preload_registry_size": len(self.preload_registry),
            "active_processes": len(self.active_processes),
            "max_cache_size": self.max_cache_size,
            "cached_files": list(self.cache.keys()),
            "preload_registered_files": list(self.preload_registry.keys()),
            "active_process_files": list(self.active_processes.keys())
        }
    
    def process_preload_queue(self, max_concurrent_loads=None):
        """
        先行読み込み登録されているファイルの読み込みプロセスを開始する関数
        
        Args:
            max_concurrent_loads: 同時に実行する最大読み込みプロセス数（Noneの場合は制限なし）
        """
        # 現在の進行中プロセス数
        current_processes = len(self.active_processes)
        
        # 同時実行数の制限
        if max_concurrent_loads is not None and current_processes >= max_concurrent_loads:
            #logging.info(f"Maximum concurrent loads ({max_concurrent_loads}) reached. Waiting for processes to complete.")
            return
        
        # 利用可能なスロット数を計算
        available_slots = float('inf') if max_concurrent_loads is None else max_concurrent_loads - current_processes
        
        # キャッシュの空き容量を確認
        available_cache_slots = self.max_cache_size - (len(self.cache) + current_processes)
        
        # いっぱいなら古いものから削除
        while available_cache_slots <= 0:
            file_to_delete = list(self.cache)[:1]
            delete_cache(self.cache, file_to_delete)
            available_cache_slots += 1

        # 実際に開始できるプロセス数（キャッシュ容量と同時実行数の少ない方）
        processes_to_start = min(available_slots, available_cache_slots, len(self.preload_registry))
        
        if processes_to_start <= 0:
            # print("No available slots for new loading processes")
            return
        
        # 先行読み込み登録から指定数のファイルを取得して読み込みを開始
        files_to_load = list(self.preload_registry.keys())[:-int(processes_to_start)]
        
        for file_path in files_to_load:
            if file_path not in self.active_processes:  # 既に進行中でないことを確認
                exif_data, param, imgset = self.preload_registry[file_path]
                self._start_loading_process(file_path, exif_data, param, imgset)
                print(f"Starting loading processes for {file_path}")
        
    def shutdown(self):
        """
        システムをシャットダウンする関数
        """
        # 全ての進行中のプロセスを終了
        #for process in self.active_processes.values():
        #    process.terminate()
        
        self.p.shutdown()
        
        logging.info("Cache system shutdown complete")
