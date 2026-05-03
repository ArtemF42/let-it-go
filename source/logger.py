from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4


class DummyLogger:
    def report_table(
        self,
        title: str,
        series: str,
        iteration: int,
        table_plot: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        print(f'[Task.logger.report_table] title={title!r}, series={series!r}, iteration={iteration}')  # fmt: skip
        print(table_plot)

    def report_single_value(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        print(f'[Task.logger.report_single_value] {name}: {value}')


class Task:
    _current_task: ClassVar[Task | None] = None

    def __init__(self, project_name: str | None = None, task_name: str | None = None) -> None:
        self.project_name = project_name
        self.task_name = task_name
        self.id = uuid4().hex
        self._logger = DummyLogger()

    @classmethod
    def init(
        cls,
        project_name: str | None = None,
        task_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Task:
        task = cls(project_name=project_name, task_name=task_name)
        cls._current_task = task
        print(
            '[Task.init] '
            f'project_name={project_name!r}, task_name={task_name!r}, '
            f'args={args!r}, kwargs={kwargs!r}, id={task.id!r}'
        )
        return task

    @classmethod
    def current_task(cls) -> Task:
        if cls._current_task is None:
            cls._current_task = cls()
            print(f'[Task.current_task] created default task id={cls._current_task.id!r}')
        return cls._current_task

    def connect(self, mutable: Any, *args: Any, **kwargs: Any) -> Any:
        print(f'[Task.connect] mutable={mutable!r}, args={args!r}, kwargs={kwargs!r}')
        return mutable

    def get_logger(self, *args: Any, **kwargs: Any) -> DummyLogger:
        print(f'[Task.get_logger] args={args!r}, kwargs={kwargs!r}')
        return self._logger

    def register_artifact(self, name: str, artifact: Any, *args: Any, **kwargs: Any) -> None:
        print(f'[Task.register_artifact] name={name!r}, args={args!r}, kwargs={kwargs!r}')
        print(artifact)
