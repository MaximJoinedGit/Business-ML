Итоговый проект по курсу "Машинное обучение в бизнесе"


Задача: предсказать по описанию вакансии является ли она фейком или нет (поле fraudulent). Бинарная классификация

Используемые признаки:

Geography - Страна, 
Gender - Пол, 
Tenure - Тип собственности, 
HasCrCard - Есть ли кредитная карта, 
IsActiveMember - Является ли активным клиентом, 
CreditScore - Кредитный рейтинг, 
Age - Возраст, 
Balance - Сумма на балансе, 
NumOfProducts - Количество продуктов компании, которыми пользуется, 
EstimatedSalary - Предполагаемая ЗП.

Задача - предсказать уйдет ли клиент от компании или нет.

Модель: Градиентный бустинг

Клонируем репозиторий и создаем образ

$ git clone https://github.com/MaximJoinedGit/Business-ML/tree/main/CR.git
$ cd CR
$ docker build . -t course_project

Запускаем контейнер

Создаем каталог локально и сохраняем туда предобученную модель (<your_local_path_to_pretrained_models> - полный путь к каталогу)

$ docker run -d -p 8180:8180 -v <your_local_path_to_pretrained_models>:/app/app/models/ course_project

Переходим на localhost:8180

Далее, используя функцию в файле CR_pred.ipynb, передаем ей параметры на вход и получаем предсказанные значения.
