Logging

Why not `print()`: It's messy, unstructured, difficult to traceback and classify log types.

The logging functions are named after the level or severity of the events they are used to track. 
**DEBUG**:  Detailed information, typically of interest only when diagnosing problems.

**INFO**: Confirmation that things are working as expected.

**WARNING**:  An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.

**ERROR**: Due to a more serious problem, the software has not been able to perform some function.

**CRITICAL**: A serious error, indicating that the program itself may be unable to continue running.


### Logging from multiple modules

If your program consists of multiple modules, here’s an example of how you could organize logging in it:


{% highlight python %}
# myapp.py
import logging
import mylib

def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.info('Started')
    mylib.do_something()
    logging.info('Finished')

if __name__ == '__main__':
    main()
{% endhighlight %}

{% highlight python %}
# mylib.py
import logging

def do_something():
    logging.info('Doing something')
{% endhighlight %}


If you run myapp.py, you should see this in myapp.log:

{% highlight python %}
INFO:root:Started
INFO:root:Doing something
INFO:root:Finished
{% endhighlight %}

{% highlight python %}
import logging
import sys

file_handler = logging.FileHandler(filename='tmp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('LOGGER_NAME')
{% endhighlight %}