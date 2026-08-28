# -*- coding: utf-8 -*-
import abc


class DataServer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start_async(self) -> None:
        pass
