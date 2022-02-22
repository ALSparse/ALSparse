.PNONY : lib
ifeq ($(ARM_ON),1)
SUBDIRS += arm
endif

ifeq ($(HIP_ON),1)
SUBDIRS += dcu
endif
ifeq ($(HYGON_ON),1)
SUBDIRS += hygon
endif
lib :
	@for d in $(SUBDIRS) ; \
	do if test -d $$d; then \
	  $(MAKE) -C $$d $(@F) || exit 1 ; \
	fi; \
	done